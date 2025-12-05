# Self-Hosted CUA Kasm Desktop on AWS (EC2 + Docker)

Goal: run the `trycua/cua-ubuntu` Kasm desktop in AWS, with:

- Docker on an Ubuntu EC2 instance
- Cua Computer Server exposed on port 8000
- Kasm desktop reachable via SSH tunnel on port 8006
- VNC/Kasm password set automatically at container start

This assumes:

- You have the AWS CLI installed and configured (`aws configure` done)
- You’re using a region like `us-west-2`
- You have (or will create) an EC2 key pair (`.pem` file)

---

## 0. Environment setup

Pick your region and export it so the commands don’t surprise you:

```bash
export AWS_REGION=us-west-2
aws configure set region $AWS_REGION
````

---

## 1. Get an Ubuntu 22.04 (Jammy) AMD64 AMI

Use the official Canonical images (owner `099720109477`):

```bash
export AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=architecture,Values=x86_64" \
  --query 'Images | sort_by(@, &CreationDate)[-1].ImageId' \
  --output text)

echo $AMI_ID
```

You only need to redo this if you care about using the latest patch-level image.

---

## 2. VPC + subnet

You must launch EC2 instances into a subnet.

### 2.1 Find your default VPC

```bash
export VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' \
  --output text)

echo $VPC_ID
```

### 2.2 Check if you already have a subnet

You can reuse an existing subnet if you have one already.

List subnets for the VPC:

```bash
aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'Subnets[].{Id:SubnetId,Cidr:CidrBlock,Az:AvailabilityZone}' \
  --output table
```

If you see subnets listed, pick one and export it:

```bash
export SUBNET_ID=subnet-xxxxxxxxxxxxxxxxx  # use your subnet ID from the table above
```

### 2.3 If you **don’t** have a subnet (one-time setup)

Only do this if the previous command returned no subnets.

Get the VPC CIDR:

```bash
export VPC_CIDR=$(aws ec2 describe-vpcs \
  --vpc-ids $VPC_ID \
  --query 'Vpcs[0].CidrBlock' \
  --output text)

echo $VPC_CIDR
```

Get an availability zone:

```bash
export AZ=$(aws ec2 describe-availability-zones \
  --query 'AvailabilityZones[0].ZoneName' \
  --output text)

echo $AZ
```

For a default VPC (`172.31.0.0/16`), this is a reasonable subnet:

```bash
aws ec2 create-subnet \
  --vpc-id $VPC_ID \
  --cidr-block 172.31.0.0/20 \
  --availability-zone $AZ
```

Then:

```bash
export SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'Subnets[0].SubnetId' \
  --output text)

echo $SUBNET_ID
```

### 2.4 Make sure instances in the subnet get public IPs

So you can SSH directly (if you later move to VPN-only, you can skip this).

```bash
aws ec2 modify-subnet-attribute \
  --subnet-id $SUBNET_ID \
  --map-public-ip-on-launch
```

---

## 3. Security group (one-time per SG)

You need a security group that at minimum allows SSH. We’ll **only expose port 22** and use SSH tunnels for everything else.

### 3.1 Create the SG

```bash
aws ec2 create-security-group \
  --group-name cua-kasm-sg \
  --description "Security group for CUA Kasm desktop" \
  --vpc-id $VPC_ID
```

Grab the ID:

```bash
export SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=cua-kasm-sg" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

echo $SG_ID
```

If this SG already exists from a previous run, just reuse its ID and skip the `create-security-group` step.

### 3.2 Allow SSH from anywhere (or lock down if you want)

For a dev box, SSH from `0.0.0.0/0` with key-based auth is common. If you want tighter control, you can replace `0.0.0.0/0` with your current IP `/32`.

```bash
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0
```

We deliberately **do not** open 8000/8006 here; those stay reachable only via SSH tunnel.

---

## 4. EC2 key pair (one-time per key)

If you already have a key pair in this region that you like, reuse it; just make sure you know the name and path to the `.pem`.

To create a new one:

```bash
aws ec2 create-key-pair \
  --key-name cua-key-pair \
  --query 'KeyMaterial' \
  --output text > cua-key-pair.pem

chmod 400 cua-key-pair.pem
```

You’ll use `--key-name cua-key-pair` when launching the instance, and `cua-key-pair.pem` locally for SSH.

---

## 5. Launch the EC2 instance

We’ll use something with enough RAM for a full desktop; `t2.medium` or `t3.medium` are okay starters.

```bash
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t2.medium \
  --key-name cua-key-pair \
  --security-group-ids $SG_ID \
  --subnet-id $SUBNET_ID \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cua-kasm}]' \
  --count 1
```

Get the instance ID:

```bash
export INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=cua-kasm" "Name=instance-state-name,Values=pending,running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

echo $INSTANCE_ID
```

Wait until it’s running:

```bash
aws ec2 wait instance-running --instance-ids $INSTANCE_ID
```

Get the public IP:

```bash
export PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo $PUBLIC_IP
```

---

## 6. SSH into the instance

On your laptop:

```bash
ssh -i /path/to/cua-key-pair.pem ubuntu@$PUBLIC_IP
```

If you get:

> Permissions 0644 for '...pem' are too open

run this on your laptop:

```bash
chmod 400 /path/to/cua-key-pair.pem
```

and try again.

---

## 7. Install Docker on the EC2 instance

On the EC2 box (logged in as `ubuntu`):

```bash
sudo apt-get update -y
sudo apt-get install -y docker.io

sudo systemctl enable --now docker

sudo docker ps
```

The last command should show an empty container list, not “command not found”.

If you want to avoid prefixing `sudo` for Docker and eliminate permission warnings on login:

```bash
sudo usermod -aG docker $USER
exit
```

Then SSH back in and `docker ps` should work without sudo. This is optional; you can just keep using `sudo docker`.

---

## 8. Pull the CUA Kasm image (amd64 tag)

Docker Hub’s `trycua/cua-ubuntu:latest` is currently `linux/arm64` only. On an amd64 instance (`uname -m` = `x86_64`), that can result in:

> no matching manifest for linux/amd64 in the manifest list entries

Docker Hub shows several amd64 tags for this image. For example:

* `trycua/cua-ubuntu:480fdceb1c2ffb6e88c6a338832a7a115c76059c` (amd64)
* `trycua/cua-ubuntu:8f4ece5dd9faf8f91b5989e2a796bcb9778700b5` (amd64)
* etc.

On the EC2 instance, pull a **known amd64 tag** instead of `latest`:

```bash
sudo docker pull trycua/cua-ubuntu:480fdceb1c2ffb6e88c6a338832a7a115c76059c
```

If this ever breaks in the future, you can switch to a newer amd64 tag from Docker Hub; the pattern is the same.

Optional but nice: retag it locally to something shorter:

```bash
sudo docker tag \
  trycua/cua-ubuntu:480fdceb1c2ffb6e88c6a338832a7a115c76059c \
  cua-ubuntu:latest
```

Then you can just refer to `cua-ubuntu:latest` locally.

---

## 9. Run the container (with VNC password set at start)

This is where we hook up:

* Cua Computer Server on host port 8000
* KasmVNC web UI on host port 8006
* VNC/Kasm password via env var

Kasm-based images support setting the VNC password via the `VNC_PW` environment variable (this is how Kasm’s own images are configured).

On the EC2 box, choose a VNC password (min 6 chars) and run:

```bash
export VNC_PASSWORD='your-strong-password'
```

Then start the container:

```bash
sudo docker run -d \
  --name cua-kasm \
  --restart unless-stopped \
  --shm-size=2g \
  -e VNC_PW="$VNC_PASSWORD" \
  -p 8000:8000 \
  -p 8006:6901 \
  cua-ubuntu:latest
```

Notes:

* Port `8000` (host) → `8000` (container): Cua Computer Server WebSocket API (`/ws`)
* Port `8006` (host) → `6901` (container): KasmVNC web UI

Verify:

```bash
sudo docker ps
```

You should see `cua-kasm` with `0.0.0.0:8000->8000` and `0.0.0.0:8006->6901`.

From the EC2 instance, confirm 8006 responds:

```bash
curl -v http://localhost:8006
```

You should see `HTTP/1.1 401 Unauthorized` with `WWW-Authenticate: Basic realm="Websockify"` – that means the KasmVNC HTTP endpoint is up and asking for Basic auth.

To sanity-check the username/password directly from the EC2 host, you can do:

```bash
curl -v http://kasm_user:$VNC_PASSWORD@localhost:8006
```

That should give you a non-401 response (e.g. `200` or a redirect), confirming `kasm_user` + `VNC_PW` are accepted.

---

## 10. SSH port forwarding from your laptop

To avoid exposing 8000/8006 to the internet at all, use SSH port forwarding.

On your laptop:

```bash
ssh -i /path/to/cua-key-pair.pem \
  -L 8000:localhost:8000 \
  -L 8006:localhost:8006 \
  ubuntu@$PUBLIC_IP
```

Keep that SSH session open while you work.

While it’s open:

* From your laptop, `http://localhost:8006` goes to the Kasm desktop UI.
* From your laptop, `ws://localhost:8000/ws` goes to the Cua Computer Server.

---

## 11. Logging into the Kasm desktop

On your laptop, with the tunnel active:

1. Open a fresh browser window (private/incognito helps if you’ve cached bad credentials).

2. Visit:

   ```text
   http://localhost:8006
   ```

3. For HTTP Basic auth / Kasm login:

   * Username: `kasm_user`
   * Password: the value you used in `VNC_PW`

If your browser is stubborn with the prompt, you can force credentials in the URL:

```text
http://kasm_user:YOUR_VNC_PASSWORD@localhost:8006
```

Once logged in, you should see the Ubuntu desktop streamed from the container.

---

## 12. Talking to the Computer Server from your code

With the SSH tunnel still active, from your laptop you can connect to the Computer Server WebSocket endpoint at:

```text
ws://localhost:8000/ws
```

Example minimal Python test:

```python
import asyncio
import json
import websockets

async def main():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"command": "version", "params": {}}))
        print("version:", await ws.recv())

        await ws.send(json.dumps({"command": "get_screen_size", "params": {}}))
        print("screen size:", await ws.recv())

asyncio.run(main())
```

You can point your actual agent at the same WebSocket URL.

---

## 13. Tearing down / avoiding surprise bills

When you’re done with the box:

Stop the instance (keeps the root volume, loses the random public IP):

```bash
aws ec2 stop-instances --instance-ids $INSTANCE_ID
```

Or terminate it (destroys the root volume):

```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

Because we used auto-assigned public IPv4, you don’t pay for the IP when the instance is stopped or terminated; you just pay for running hours + storage.

---

## Summary of “one-time vs every time”

**One-time:**

* Create security group `cua-kasm-sg` and set SSH ingress
* Create subnet if your default VPC has none
* Create key pair `cua-key-pair` and store `.pem` securely

**Every time you want a new remote desktop:**

* Get or reuse `AMI_ID` for Ubuntu 22.04
* Launch `run-instances` into `$SUBNET_ID` with `$SG_ID` and `cua-key-pair`
* SSH in, install Docker (`docker.io`) if this is a new instance
* Pull `trycua/cua-ubuntu:<amd64-tag>` (or reuse local image)
* Run container with `VNC_PW`, `-p 8000:8000`, `-p 8006:6901`
* SSH tunnel ports 8000 and 8006 from your laptop
* Login to Kasm as `kasm_user` with the `VNC_PW` password you set

# Websockets

* The **VNC/Kasm HTTP UI** *is* password-protected (HTTP Basic).
* The **Computer Server WebSocket (`/ws` on port 8000)** is, by default, effectively **unauthenticated** – it trusts anyone who can reach the port.
* In our setup, security is coming from **network isolation + SSH**. If you keep it that way, you’re okay. If you ever expose 8000 directly, you **must** add auth in front.


## 1. Does the WS endpoint have auth?

By default: **no meaningful auth**.

The Cua Computer Server (the thing on `ws://host:8000/ws`) is designed to be used behind some trusted boundary. The protocol is:

* WebSocket:

  * You connect to `/ws`
  * You send JSON like `{"command": "version", "params": {}}`
  * It replies JSON with `success`, etc.

There’s no built-in username/password or token check in the core examples; they assume that whoever can open the socket is already trusted.

So if you put `ws://PUBLIC_IP:8000/ws` straight on the internet with no SSH/proxy/etc, anyone who finds it can:

* Move the mouse
* Type
* Click
* Open apps
* Potentially exfiltrate data from that VM

## 2. In *our* setup, what’s protecting it?

Right now, we have this:

* Security Group: **only port 22 is open** to the world.

* We start SSH from your laptop with:

  ```bash
  ssh -i ~/Downloads/cua-key-pair.pem \
    -L 8000:localhost:8000 \
    -L 8006:localhost:8006 \
    ubuntu@<PUBLIC_IP>
  ```

* That means:

  * The Computer Server is only reachable on `localhost:8000` on your Mac (tunneled via SSH).
  * The Kasm UI is only reachable on `localhost:8006` on your Mac.

So effective protection is:

* **SSH key-based auth** on port 22.
* **No one can hit port 8000/8006 directly** from the internet because the SG doesn’t allow it.

In that model:

* It’s fine that the WS endpoint has no extra auth: it’s sitting behind SSH.
* VNC having HTTP Basic auth is “belt and suspenders” on the browser endpoint; it doesn’t change the security story for the WebSocket.

If you keep this architecture (no SG rule for 8000, always tunnel), you’re in a good place.

---

## 3. What if you *did* want to expose 8000 without SSH?

Then you really should add one (or more) of:

### Option A: Reverse proxy with auth + TLS

Put nginx / Caddy / Traefik in front of the Computer Server:

* Internet → nginx (HTTPS, Basic auth / OAuth / whatever) → Computer Server `ws://localhost:8000/ws`.

nginx example idea (not full config, just the gist):

```nginx
location /computer-ws/ {
    proxy_pass http://localhost:8000/ws;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "Upgrade";
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
}
```

Then you connect from your client to:

```text
wss://your-domain/computer-ws/
```

with HTTP Basic auth (or some other scheme). This gives you:

* TLS
* HTTP-level auth to gate the WS
* The ability to hang rate limiting / IP allow lists / etc. in front

### Option B: VPN (Tailscale / WireGuard)

Treat the EC2 host as a node in a private network:

* No public SG rules for 8000/8006.
* Instance has Tailscale, you connect laptop → Tailscale.
* Computer Server reachable at `ws://100.x.y.z:8000/ws`.
* Kasm reachable at `http://100.x.y.z:8006`.

Similar trust model to SSH tunneling, just ergonomic in a different way.

### Option C: Application-level auth in your own “agent gateway”

If you’re building a service, you can:

* Run a small gateway process that exposes its own authenticated HTTP/WebSocket API to clients (with tokens, JWTs, etc.).
* That gateway then talks to the Computer Server on `localhost:8000`.

This keeps Cua’s server “dumb” and internal; all auth/permissions live in your app.

* **You do not need additional auth on the WS endpoint right now**, because:

  * The SG only exposes 22.
  * You only reach 8000 via SSH tunnel.

If you stick with:

* SG open only on 22
* Always using `-L 8000:localhost:8000` from your laptop

then:

* The VNC authentication protects the browser UI.
* SSH protects *both* VNC and the WS endpoint.
* The WS endpoint’s lack of built-in auth is fine.

If you ever decide “I want to connect from somewhere that can’t do SSH, but can reach a public URL” → that’s when you introduce a reverse proxy + auth or a VPN.
