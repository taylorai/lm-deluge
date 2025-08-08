# import argparse
# import asyncio
# import os
# import sys
# from typing import Optional

# from .client import LLMClient
# from .models import registry, APIModel
# from .prompt import Conversation, Message


# def _ensure_api_key_for_model(model_id: str, passed_api_key: Optional[str] = None):
#     model: APIModel = APIModel.from_registry(model_id)
#     env_var = model.api_key_env_var or ""
#     if not env_var:
#         return  # Some providers (e.g., Bedrock entries) don't use a single key
#     if os.getenv(env_var):
#         return
#     if passed_api_key:
#         os.environ[env_var] = passed_api_key
#         return
#     # If we get here, interactive prompting should occur at the UI layer.
#     # In non-interactive contexts, we will error before calling this without key.


# def run_non_interactive(model_id: str, prompt_text: str, api_key: Optional[str]):
#     _ensure_api_key_for_model(model_id, api_key)
#     client = LLMClient(model_names=[model_id], progress="manual")
#     # Single round, print completion only to stdout
#     completions = asyncio.run(
#         client.process_prompts_async(
#             [Conversation.user(prompt_text)],
#             return_completions_only=True,
#             show_progress=False,
#         )
#     )
#     out = completions[0] if completions and completions[0] is not None else ""
#     # Write raw completion to stdout with no extra decoration
#     sys.stdout.write(out)
#     if out and not out.endswith("\n"):
#         sys.stdout.write("\n")


# # -------- Textual UI (interactive chat) --------
# try:
#     from textual.app import App, ComposeResult
#     from textual.containers import Container, Horizontal
#     from textual.widgets import Footer, Header, Input, Static, Button, ListView, ListItem, Label
#     from textual.widgets._rich_log import RichLog
#     from textual.reactive import reactive
#     TEXTUAL_AVAILABLE = True
# except Exception:  # pragma: no cover - textual may not be installed in some dev envs
#     TEXTUAL_AVAILABLE = False


# if TEXTUAL_AVAILABLE:
#     class ModelPicker(Static):
#         """Minimal model picker: arrows to move, Enter to select."""

#         def __init__(self, preselected: Optional[str] = None):
#             super().__init__()
#             self.preselected = preselected

#         def compose(self) -> ComposeResult:  # type: ignore[override]
#             # Keep it terminal-y: one-line hint + list. No buttons.
#             yield Static("Pick a model (Enter)", classes="hint")
#             list_items: list[ListItem] = []
#             # Curated small set to avoid scrollbars
#             preferred = [
#                 "gpt-5",
#                 "gpt-5-chat",
#                 "gpt-5-mini",
#                 "claude-4-sonnet",
#                 "gemini-2.5-pro",
#                 "gemini-2.5-flash",
#                 "gemini-2.0-flash",
#             ]
#             for mid in preferred:
#                 if mid in registry:
#                     list_items.append(ListItem(Label(mid)))
#             yield ListView(*list_items, classes="model-list")

#         def on_mount(self) -> None:  # type: ignore[override]
#             # Focus the list so Enter works immediately
#             self.query_one(ListView).focus()

#         def get_selected(self) -> Optional[str]:
#             listview = self.query_one(ListView)
#             if not listview.index is None and 0 <= listview.index < len(listview.children):
#                 label = listview.children[listview.index].query_one(Label)
#                 return label.renderable if isinstance(label.renderable, str) else str(label.renderable)
#             return None

#         def on_key(self, event):  # type: ignore[override]
#             # Select current item on Enter
#             try:
#                 key = getattr(event, "key", None)
#             except Exception:
#                 key = None
#             if key == "enter":
#                 sel = self.get_selected()
#                 if sel:
#                     # Ask app to proceed with the chosen model
#                     getattr(self.app, "model_chosen", lambda *_: None)(sel)  # type: ignore[attr-defined]


#     class ApiKeyPrompt(Static):
#         def __init__(self, env_var: str):
#             super().__init__()
#             self.env_var = env_var
#             self.input = Input(password=True, placeholder=f"Enter {env_var}")

#         def compose(self) -> ComposeResult:  # type: ignore[override]
#             yield Static(f"API key required: set {self.env_var}", classes="title")
#             yield self.input
#             yield Button("Save", id="save-key", variant="primary")

#         def value(self) -> str:
#             return self.input.value


#     class MessagesView(RichLog):
#         def __init__(self, **kwargs):
#             # Terminal-like log with markup and auto-scroll
#             super().__init__(wrap=True, markup=True, auto_scroll=True, **kwargs)

#         def append_user(self, text: str):
#             self.write(f"[bold cyan]You:[/bold cyan] {text}")

#         def append_assistant(self, text: str):
#             self.write(f"[bold magenta]Model:[/bold magenta] {text}")


#     class ChatInput(Horizontal):
#         def compose(self) -> ComposeResult:  # type: ignore[override]
#             self.input = Input(placeholder="Type message, Enter to send")
#             yield self.input


#     class DelugeApp(App):
#         CSS = """
#         #screen { height: 100%; }
#         .chat { height: 1fr; padding: 0 1; }
#         .composer { dock: bottom; height: 3; }
#         """

#         BINDINGS = [
#             ("ctrl+c", "quit", "Quit"),
#         ]

#         model_id = reactive("")
#         api_env_var = reactive("")

#         def __init__(self, model_arg: Optional[str], api_key_arg: Optional[str]):
#             super().__init__()
#             self._model_arg = model_arg
#             self._api_key_arg = api_key_arg
#             self._conversation = Conversation.system("You are a helpful assistant.")
#             self._client = None

#         def compose(self) -> ComposeResult:  # type: ignore[override]
#             yield Header(show_clock=True)
#             self.body = Container(id="screen")
#             yield self.body
#             yield Footer()

#         def on_mount(self):  # type: ignore[override]
#             # Step 1: pick model if not provided
#             if not self._model_arg:
#                 self.model_picker = ModelPicker()
#                 self.body.mount(self.model_picker)
#             else:
#                 self.model_id = self._model_arg
#                 self._after_model_selected()

#         def action_quit(self) -> None:  # type: ignore[override]
#             self.exit()

#         def _after_model_selected(self):
#             # Resolve API requirement
#             model = APIModel.from_registry(self.model_id)
#             self.api_env_var = model.api_key_env_var or ""
#             if self.api_env_var and not os.getenv(self.api_env_var):
#                 if self._api_key_arg:
#                     os.environ[self.api_env_var] = self._api_key_arg
#                     self._show_chat()
#                 else:
#                     # Prompt for key
#                     self.body.remove_children()
#                     self.key_prompt = ApiKeyPrompt(self.api_env_var)
#                     self.body.mount(self.key_prompt)
#             else:
#                 self._show_chat()

#         def model_chosen(self, sel: str) -> None:
#             """Called by ModelPicker when Enter is pressed on a selection."""
#             self.model_id = sel
#             self._after_model_selected()

#         def _show_chat(self):
#             self.body.remove_children()
#             # Build UI
#             self.messages = MessagesView(classes="chat")
#             self.composer = ChatInput(classes="composer")
#             self.body.mount(self.messages)
#             self.body.mount(self.composer)
#             # Focus input after mounting
#             self.set_focus(self.composer.input)
#             # Init client
#             self._client = LLMClient(model_names=[self.model_id], progress="manual")
#             # Update header subtitle
#             self.query_one(Header).sub_title = f"Model: {self.model_id}"

#         async def _send_and_receive(self, text: str):
#             # Append user message
#             self._conversation.add(Message.user(text))
#             self.messages.append_user(text)
#             # Call model (non-streaming for simplicity across providers)
#             responses = await self._client.process_prompts_async(
#                 [self._conversation], return_completions_only=False, show_progress=False
#             )
#             resp = responses[0]
#             if resp and resp.completion:
#                 self._conversation.add(Message.ai(resp.completion))
#                 self.messages.append_assistant(resp.completion)
#             else:
#                 self.messages.append_assistant("<no response>")

#         async def on_button_pressed(self, event):  # type: ignore[override]
#             if hasattr(event.button, "id"):
#                 if event.button.id == "save-key":
#                     key = self.key_prompt.value().strip()
#                     if self.api_env_var and key:
#                         os.environ[self.api_env_var] = key
#                     self._show_chat()
#                 elif event.button.id == "send":
#                     text = self.composer.input.value.strip()
#                     if text:
#                         self.composer.input.value = ""
#                         await self._send_and_receive(text)

#         async def on_input_submitted(self, event: Input.Submitted):  # type: ignore[override]
#             if isinstance(event.input.parent, ChatInput):
#                 text = event.value.strip()
#                 if text:
#                     self.composer.input.value = ""
#                     await self._send_and_receive(text)


# def run_interactive(model: Optional[str], api_key: Optional[str]):
#     if not TEXTUAL_AVAILABLE:
#         sys.stderr.write(
#             "Textual is not installed. Please install with `pip install textual` or reinstall lm_deluge.\n"
#         )
#         sys.exit(2)
#     app = DelugeApp(model, api_key)  # type: ignore[name-defined]
#     app.run()


# def main():
#     parser = argparse.ArgumentParser(prog="deluge", description="Deluge CLI")
#     parser.add_argument("prompt", nargs="*", help="Prompt text (non-interactive -p only)")
#     parser.add_argument("--model", dest="model", help="Model ID to use")
#     parser.add_argument("--api-key", dest="api_key", help="API key for chosen model provider")
#     parser.add_argument(
#         "-p",
#         dest="print_mode",
#         action="store_true",
#         help="Print single completion to stdout (non-interactive)",
#     )

#     args = parser.parse_args()

#     if args.print_mode:
#         # Determine prompt text
#         prompt_text = " ".join(args.prompt).strip()
#         if not prompt_text and not sys.stdin.isatty():
#             prompt_text = sys.stdin.read()
#         if not prompt_text:
#             sys.stderr.write("No prompt provided. Pass text or pipe input.\n")
#             sys.exit(2)

#         # Determine model
#         model_id = args.model or os.getenv("DELUGE_DEFAULT_MODEL") or "gpt-4o-mini"
#         # Require API key non-interactively if provider needs it and not set
#         env_var = APIModel.from_registry(model_id).api_key_env_var or ""
#         if env_var and not (os.getenv(env_var) or args.api_key):
#             sys.stderr.write(
#                 f"Missing API key. Set {env_var} or pass --api-key.\n"
#             )
#             sys.exit(2)
#         run_non_interactive(model_id, prompt_text, args.api_key)
#         return

#     # Interactive Textual chat
#     run_interactive(args.model, args.api_key)


# if __name__ == "__main__":
#     main()
