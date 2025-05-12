shared utility for making openai & other llm calls so that we don't have to reinvent the wheel for every repo


# TODO: handle ClientOS error
#         The ClientOSError: [Errno 32] Broken pipe error in aiohttp usually indicates that the connection was abruptly closed. This can happen for a variety of reasons, including network issues, server-side problems, or issues in your client code.

# Here are some steps to help you diagnose and fix the issue:

#     Retry Logic: Implement a retry mechanism to handle transient network issues.
#     Connection Timeout: Ensure that you have a reasonable timeout set for your requests.
#     Session Management: Properly manage the aiohttp.ClientSession to ensure that connections are properly closed and reused.
#     Exception Handling: Add robust exception handling to capture and handle ClientOSError.

# Here's an example demonstrating these steps:

# python

# import aiohttp
# import asyncio
# import async_timeout

# async def fetch(url, session):
#     try:
#         async with async_timeout.timeout(10):
#             async with session.get(url) as response:
#                 return await response.text()
#     except aiohttp.ClientOSError as e:
#         print(f"ClientOSError occurred: {e}")
#         # Implement retry logic here if needed
#     except asyncio.TimeoutError:
#         print("Request timed out")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# async def main():
#     url = 'http://example.com'
#     async with aiohttp.ClientSession() as session:
#         response = await fetch(url, session)
#         if response:
#             print(response)

# if __name__ == '__main__':
#     asyncio.run(main())

# Key Points:

#     Retry Logic: In the fetch function, you can add a loop to retry the request a certain number of times before giving up.
#     Connection Timeout: Using async_timeout to ensure that requests do not hang indefinitely.
#     Session Management: Creating a single ClientSession per application and reusing it across multiple requests.
#     Exception Handling: Catching ClientOSError and other exceptions to handle them appropriately.

# By following these steps, you can mitigate and handle the Broken pipe error more effectively.
