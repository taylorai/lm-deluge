CLAUDE_3_HAIKU_US_SOURCE_REGIONS = [
    "us-east-1",
    "us-west-2",
]

CLAUDE_4_SONNET_US_SOURCE_REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
]

CLAUDE_4_OPUS_US_SOURCE_REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-2",
]

CLAUDE_4_5_US_SOURCE_REGIONS = [
    "ca-central-1",
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
]

CLAUDE_4_6_US_SOURCE_REGIONS = [
    "ca-central-1",
    "ca-west-1",
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
]

# Source regions for global cross-region profiles as documented by AWS Bedrock.
# Global profile routing and supported regions can evolve over time.
CLAUDE_GLOBAL_SOURCE_REGIONS_V45 = [
    "af-south-1",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ca-central-1",
    "eu-central-1",
    "eu-central-2",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "il-central-1",
    # "me-central-1", -- this one got bombed
    "me-south-1",
    "sa-east-1",
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
]

CLAUDE_GLOBAL_SOURCE_REGIONS_V46 = [
    "af-south-1",
    "ap-east-2",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ap-southeast-5",
    "ap-southeast-7",
    "ca-central-1",
    "ca-west-1",
    "eu-central-1",
    "eu-central-2",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "il-central-1",
    # "me-central-1", -- this one got bombed
    "me-south-1",
    "mx-central-1",
    "sa-east-1",
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
]

CLAUDE_4_SONNET_GLOBAL_SOURCE_REGIONS = [
    "ap-northeast-1",
    "eu-west-1",
    "us-east-1",
    "us-east-2",
    "us-west-2",
]

NOVA_US_SOURCE_REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-2",
]


BEDROCK_MODELS = {
    #  ███████████               █████                             █████
    # ░░███░░░░░███             ░░███                             ░░███
    #  ░███    ░███  ██████   ███████  ████████   ██████   ██████  ░███ █████
    #  ░██████████  ███░░███ ███░░███ ░░███░░███ ███░░███ ███░░███ ░███░░███
    #  ░███░░░░░███░███████ ░███ ░███  ░███ ░░░ ░███ ░███░███ ░░░  ░██████░
    #  ░███    ░███░███░░░  ░███ ░███  ░███     ░███ ░███░███  ███ ░███░░███
    #  ███████████ ░░██████ ░░████████ █████    ░░██████ ░░██████  ████ █████
    # ░░░░░░░░░░░   ░░░░░░   ░░░░░░░░ ░░░░░      ░░░░░░   ░░░░░░  ░░░░ ░░░░░
    "claude-3-haiku-bedrock": {
        "id": "claude-3-haiku-bedrock",
        "name": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "regions": CLAUDE_3_HAIKU_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 0.25,
        "output_cost": 1.25,
        "supports_images": True,
    },
    "claude-4-sonnet-bedrock": {
        "id": "claude-4-sonnet-bedrock",
        "name": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "regions": CLAUDE_4_SONNET_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "reasoning_model": True,
        "supports_images": True,
    },
    "claude-4.1-opus-bedrock": {
        "id": "claude-4.1-opus-bedrock",
        "name": "us.anthropic.claude-opus-4-1-20250805-v1:0",
        "regions": CLAUDE_4_OPUS_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 15.0,
        "output_cost": 75.0,
        "reasoning_model": True,
        "supports_images": True,
    },
    "claude-4.5-haiku-bedrock": {
        "id": "claude-4.5-haiku-bedrock",
        "name": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "regions": CLAUDE_4_5_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 0.25,
        "output_cost": 1.25,
        "reasoning_model": True,
        "supports_images": True,
    },
    "claude-4.5-sonnet-bedrock": {
        "id": "claude-4.5-sonnet-bedrock",
        "name": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "regions": CLAUDE_4_5_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "reasoning_model": True,
        "supports_images": True,
    },
    "claude-4.5-opus-bedrock": {
        "id": "claude-4.5-opus-bedrock",
        "name": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "regions": CLAUDE_4_5_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 15.0,
        "output_cost": 75.0,
        "reasoning_model": True,
        "supports_images": True,
    },
    "claude-4.6-opus-bedrock": {
        "id": "claude-4.6-opus-bedrock",
        "name": "us.anthropic.claude-opus-4-6-v1",
        "regions": CLAUDE_4_6_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 5.0,
        "output_cost": 25.0,
        "supports_json": True,
        "reasoning_model": True,
    },
    "claude-4.6-sonnet-bedrock": {
        "id": "claude-4.6-sonnet-bedrock",
        "name": "us.anthropic.claude-sonnet-4-6",
        "regions": CLAUDE_4_6_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "supports_json": True,
        "reasoning_model": True,
        "supports_images": True,
    },
    "claude-4-sonnet-bedrock-global": {
        "id": "claude-4-sonnet-bedrock-global",
        "name": "global.anthropic.claude-sonnet-4-20250514-v1:0",
        "regions": CLAUDE_4_SONNET_GLOBAL_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "reasoning_model": True,
        "supports_images": True,
    },
    "claude-4.5-haiku-bedrock-global": {
        "id": "claude-4.5-haiku-bedrock-global",
        "name": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "regions": CLAUDE_GLOBAL_SOURCE_REGIONS_V45,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 1.0,
        "output_cost": 5.0,
        "reasoning_model": True,
        "supports_images": True,
    },
    "claude-4.5-sonnet-bedrock-global": {
        "id": "claude-4.5-sonnet-bedrock-global",
        "name": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "regions": CLAUDE_GLOBAL_SOURCE_REGIONS_V45,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "reasoning_model": True,
        "supports_images": True,
    },
    "claude-4.5-opus-bedrock-global": {
        "id": "claude-4.5-opus-bedrock-global",
        "name": "global.anthropic.claude-opus-4-5-20251101-v1:0",
        "regions": CLAUDE_GLOBAL_SOURCE_REGIONS_V45,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 5.0,
        "output_cost": 25.0,
        "reasoning_model": True,
        "supports_images": True,
    },
    "claude-4.6-opus-bedrock-global": {
        "id": "claude-4.6-opus-bedrock-global",
        "name": "global.anthropic.claude-opus-4-6-v1",
        "regions": CLAUDE_GLOBAL_SOURCE_REGIONS_V46,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 5.0,
        "output_cost": 25.0,
        "supports_json": True,
        "reasoning_model": True,
    },
    "claude-4.6-sonnet-bedrock-global": {
        "id": "claude-4.6-sonnet-bedrock-global",
        "name": "global.anthropic.claude-sonnet-4-6",
        "regions": CLAUDE_GLOBAL_SOURCE_REGIONS_V46,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "supports_json": True,
        "reasoning_model": True,
        "supports_images": True,
    },
    # GPT-OSS on AWS Bedrock
    "gpt-oss-120b-bedrock": {
        "id": "gpt-oss-120b-bedrock",
        "name": "openai.gpt-oss-120b-1:0",
        "regions": ["us-west-2"],
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "supports_json": False,
        "supports_logprobs": False,
        "supports_responses": False,
        "reasoning_model": True,
        "supports_images": False,
    },
    "gpt-oss-20b-bedrock": {
        "id": "gpt-oss-20b-bedrock",
        "name": "openai.gpt-oss-20b-1:0",
        "regions": ["us-west-2"],
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "supports_json": False,
        "supports_logprobs": False,
        "supports_responses": False,
        "reasoning_model": True,
        "supports_images": False,
    },
    #  ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗
    #  ████╗  ██║██╔═══██╗██║   ██║██╔══██╗
    #  ██╔██╗ ██║██║   ██║██║   ██║███████║
    #  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
    #  ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║
    #  ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝
    "nova-micro": {
        "id": "nova-micro",
        "name": "us.amazon.nova-micro-v1:0",
        "regions": NOVA_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock-nova",
        "input_cost": 0.035,
        "output_cost": 0.14,
        "supports_images": False,
    },
    "nova-lite": {
        "id": "nova-lite",
        "name": "us.amazon.nova-lite-v1:0",
        "regions": NOVA_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock-nova",
        "input_cost": 0.06,
        "output_cost": 0.24,
        "supports_images": True,
    },
    "nova-pro": {
        "id": "nova-pro",
        "name": "us.amazon.nova-pro-v1:0",
        "regions": NOVA_US_SOURCE_REGIONS,
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock-nova",
        "input_cost": 0.80,
        "output_cost": 3.20,
        "supports_images": True,
    },
}

# | Model | Source regions in lm-deluge | RPM / source region | TPM / source region | Approx sprayed RPM | Approx sprayed TPM |
# |---|---:|---:|---:|---:|---:|
# | claude-4-sonnet-bedrock-global | 5 | 200 | 200,000 | 1,000 | 1,000,000 |
# | claude-4.5-haiku-bedrock-global | 27 | 1,000 | 5,000,000 | 27,000 | 135,000,000 |
# | claude-4.5-sonnet-bedrock-global | 27 | 1,000 | 5,000,000 | 27,000 | 135,000,000 |
# | claude-4.5-opus-bedrock-global | 27 | 500 | 2,000,000 | 13,500 | 54,000,000 |
# | claude-4.6-opus-bedrock-global | 32 | 500 | 2,000,000 | 16,000 | 64,000,000 |
# | claude-4.6-sonnet-bedrock-global | 32 | 10,000 | 5,000,000 | 320,000 | 160,000,000 |
