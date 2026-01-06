"""Live test for PhilipsHueManager - lists lights and turns a reachable one green."""

import asyncio
import json

from dotenv import load_dotenv

from lm_deluge.tool.prefab import PhilipsHueManager

load_dotenv()


async def main():
    manager = PhilipsHueManager()
    tools = manager.get_tools()

    # Get tools by name
    list_tool = next(t for t in tools if t.name == "hue_list_lights")
    color_tool = next(t for t in tools if t.name == "hue_set_color")

    # List all lights
    print("Listing lights...")
    result = await list_tool.run()
    data = json.loads(result)

    if data["status"] != "success":
        print(f"Error: {data.get('error')}")
        return

    lights = data["lights"]
    print(f"Found {len(lights)} lights:\n")
    for light in lights:
        status = "ON" if light["on"] else "OFF"
        reachable = "reachable" if light["reachable"] else "unreachable"
        print(
            f"  {light['id']}: {light['name']} - {status}, {reachable}, brightness={light['brightness']}"
        )

    # Find first reachable light
    reachable_light = next((lt for lt in lights if lt["reachable"]), None)

    if not reachable_light:
        print("\nNo reachable lights found!")
        return

    print(
        f"\nTurning light {reachable_light['id']} ({reachable_light['name']}) green..."
    )
    result = await color_tool.run(
        light_id=reachable_light["id"], color="green", brightness=200
    )
    data = json.loads(result)

    if data["status"] == "success":
        print("Success!")
        for change in data.get("changes", []):
            print(f"  {change}")
    else:
        print(f"Error: {data.get('error') or data.get('errors')}")


if __name__ == "__main__":
    asyncio.run(main())
