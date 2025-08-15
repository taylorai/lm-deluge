DEEPSEEK_MODELS = {
    #  ______                                     _
    # (______)                                   | |
    #  _     _ _____ _____ ____   ___ _____ _____| |  _
    # | |   | | ___ | ___ |  _ \ /___) ___ | ___ | |_/ )
    # | |__/ /| ____| ____| |_| |___ | ____| ____|  _ (
    # |_____/ |_____)_____)  __/(___/|_____)_____)_| \_)
    #                     |_|
    "deepseek-chat": {
        "id": "deepseek-chat",
        "name": "deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env_var": "DEEPSEEK_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.27,
        "output_cost": 1.10,
    },
    "deepseek-r1": {
        "id": "deepseek-r1",
        "name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env_var": "DEEPSEEK_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.55,
        "output_cost": 2.19,
    },
}
