/**
 * Cloudflare Worker that exposes a thin HTTP API over the Sandbox SDK.
 *
 * Endpoints:
 *   POST /sandbox/create        - create or get a sandbox by id
 *   POST /sandbox/:id/exec      - execute a command
 *   POST /sandbox/:id/write     - write a file
 *   POST /sandbox/:id/read      - read a file
 *   POST /sandbox/:id/list      - list files in a directory
 *   POST /sandbox/:id/delete    - delete a file
 *   POST /sandbox/:id/mkdir     - create a directory
 *   POST /sandbox/:id/expose    - expose a port and get preview URL
 *   DELETE /sandbox/:id         - destroy a sandbox
 *
 * All requests require an Authorization header matching the SANDBOX_API_KEY secret.
 */

import { getSandbox, proxyToSandbox } from "@cloudflare/sandbox";

export { Sandbox } from "@cloudflare/sandbox";

interface Env {
  Sandbox: Parameters<typeof getSandbox>[0];
  SANDBOX_API_KEY: string;
}

function json(data: unknown, status = 200): Response {
  return Response.json(data, { status });
}

function err(message: string, status = 400): Response {
  return json({ error: message }, status);
}

function authorize(request: Request, env: Env): Response | null {
  const key = request.headers.get("Authorization")?.replace("Bearer ", "");
  if (!key || key !== env.SANDBOX_API_KEY) {
    return err("Unauthorized", 401);
  }
  return null;
}

/** Parse JSON body, returning null + error response on failure. */
async function body<T>(request: Request): Promise<[T | null, Response | null]> {
  try {
    const data = (await request.json()) as T;
    return [data, null];
  } catch {
    return [null, err("Invalid JSON body")];
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Preview URLs are authenticated by the short-lived token embedded in the
    // hostname and must be routed before the worker API-key check.
    const proxyResponse = await proxyToSandbox(request, env);
    if (proxyResponse) return proxyResponse;

    // Auth check
    const authErr = authorize(request, env);
    if (authErr) return authErr;

    const url = new URL(request.url);
    const path = url.pathname;

    // --- POST /sandbox/create ---
    if (path === "/sandbox/create" && request.method === "POST") {
      const [data, parseErr] = await body<{ id?: string }>(request);
      if (parseErr) return parseErr;

      const id = data?.id || crypto.randomUUID();
      const sandbox = getSandbox(env.Sandbox, id);

      // Auto-sleep after 5 minutes of inactivity (stops billing)
      await sandbox.setSleepAfter("30m");

      // Ping to ensure it's alive
      try {
        const result = await sandbox.exec("echo ok");
        return json({ id, status: "ready", ping: result.stdout?.trim() });
      } catch (e: any) {
        return json(
          { id, status: "starting", message: e?.message || "starting up" },
          202
        );
      }
    }

    // All other routes: /sandbox/:id/action
    const match = path.match(/^\/sandbox\/([^/]+)\/([^/]+)$/);
    const deleteMatch = path.match(/^\/sandbox\/([^/]+)$/);

    // --- DELETE /sandbox/:id ---
    if (deleteMatch && request.method === "DELETE") {
      const id = deleteMatch[1];
      const sandbox = getSandbox(env.Sandbox, id);
      try {
        // Kill all processes and let container idle out
        await sandbox.exec("kill -9 -1 2>/dev/null || true");
        return json({ id, status: "destroyed" });
      } catch {
        return json({ id, status: "destroyed" });
      }
    }

    if (!match) {
      return err("Not found. Use POST /sandbox/create or /sandbox/:id/:action", 404);
    }

    const [, sandboxId, action] = match;
    const sandbox = getSandbox(env.Sandbox, sandboxId);

    // --- POST /sandbox/:id/exec ---
    if (action === "exec" && request.method === "POST") {
      const [data, parseErr] = await body<{
        command: string;
        timeout?: number;
      }>(request);
      if (parseErr) return parseErr;
      if (!data?.command) return err("Missing 'command'");

      try {
        const opts: any = {};
        if (data.timeout) {
          opts.timeoutMs = data.timeout;
        }
        const result = await sandbox.exec(data.command, opts);
        return json({
          stdout: result.stdout || "",
          stderr: result.stderr || "",
          exitCode: result.exitCode,
          success: result.success,
        });
      } catch (e: any) {
        return json(
          { stdout: "", stderr: e?.message || "exec failed", exitCode: -1, success: false },
          500
        );
      }
    }

    // --- POST /sandbox/:id/write ---
    if (action === "write" && request.method === "POST") {
      const [data, parseErr] = await body<{ path: string; content: string }>(
        request
      );
      if (parseErr) return parseErr;
      if (!data?.path || data.content === undefined)
        return err("Missing 'path' or 'content'");

      await sandbox.writeFile(data.path, data.content);
      return json({ ok: true });
    }

    // --- POST /sandbox/:id/read ---
    if (action === "read" && request.method === "POST") {
      const [data, parseErr] = await body<{ path: string }>(request);
      if (parseErr) return parseErr;
      if (!data?.path) return err("Missing 'path'");

      try {
        const file = await sandbox.readFile(data.path);
        return json({ content: file.content });
      } catch (e: any) {
        return err(e?.message || "read failed", 404);
      }
    }

    // --- POST /sandbox/:id/list ---
    if (action === "list" && request.method === "POST") {
      const [data, parseErr] = await body<{ path?: string }>(request);
      if (parseErr) return parseErr;

      try {
        const files = await sandbox.listFiles(data?.path || "/workspace");
        return json({ files });
      } catch (e: any) {
        return err(e?.message || "list failed", 500);
      }
    }

    // --- POST /sandbox/:id/delete ---
    if (action === "delete" && request.method === "POST") {
      const [data, parseErr] = await body<{ path: string }>(request);
      if (parseErr) return parseErr;
      if (!data?.path) return err("Missing 'path'");

      try {
        await sandbox.deleteFile(data.path);
        return json({ ok: true });
      } catch (e: any) {
        return err(e?.message || "delete failed", 500);
      }
    }

    // --- POST /sandbox/:id/mkdir ---
    if (action === "mkdir" && request.method === "POST") {
      const [data, parseErr] = await body<{ path: string }>(request);
      if (parseErr) return parseErr;
      if (!data?.path) return err("Missing 'path'");

      try {
        await sandbox.mkdir(data.path);
        return json({ ok: true });
      } catch (e: any) {
        return err(e?.message || "mkdir failed", 500);
      }
    }

    // --- POST /sandbox/:id/expose ---
    if (action === "expose" && request.method === "POST") {
      const [data, parseErr] = await body<{ port: number }>(request);
      if (parseErr) return parseErr;
      if (!data?.port) return err("Missing 'port'");

      try {
        const result = await sandbox.exposePort(data.port, {
          hostname: url.hostname,
        });
        return json(result);
      } catch (e: any) {
        return err(e?.message || "expose failed", 500);
      }
    }

    return err(`Unknown action: ${action}`, 404);
  },
};
