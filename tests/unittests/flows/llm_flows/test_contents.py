# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.flows.llm_flows import contents
from google.adk.models import LlmRequest
from google.adk.sessions import Session
from google.genai import types
import pytest

from ... import testing_utils
from ...testing_utils import simplify_contents


greeting_text_part = types.Part.from_text(text="Add one to 5")
agent_summary_text_part = types.Part.from_text(text="The functions have been called")
continuation_text_part = types.Part.from_text(text="Go on")
fc_part_anon = types.Part.from_function_call(name="increase_by_one", args={"x": 5})
fr_part_anon = types.Part.from_function_response(name="increase_by_one", response={"result": 6})
fr_part_alt_anon = types.Part.from_function_response(name="increase_by_one", response={"result": 4})
fc_part_0 = types.Part(function_call=types.FunctionCall(id="0", name=fc_part_anon.function_call.name, args=fc_part_anon.function_call.args))
fr_part_0 = types.Part(function_response=types.FunctionResponse(id="0", name=fr_part_anon.function_response.name, response=fr_part_anon.function_response.response))
fr_part_alt_0 = types.Part(function_response=types.FunctionResponse(id="0", name=fr_part_alt_anon.function_response.name, response=fr_part_alt_anon.function_response.response))
fc_part_1 = types.Part(function_call=types.FunctionCall(id="1", name=fc_part_anon.function_call.name, args=fc_part_anon.function_call.args))
fr_part_1 = types.Part(function_response=types.FunctionResponse(id="1", name=fr_part_anon.function_response.name, response=fr_part_anon.function_response.response))

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "simple_events, simple_contents",
    [
        (
            [("user", [greeting_text_part])],
            [
                ("user", "Add one to 5"),
            ],
        ),
        (
            [
                ("user", [greeting_text_part]),
                ("agent", [fc_part_0]),
                ("user", [fr_part_0]),
            ],
            [
                ("user", "Add one to 5"),
                ("model", fc_part_anon),
                ("user", fr_part_anon),
            ],
        ),
        (
            [
                ("user", [greeting_text_part]),
                ("agent", [fc_part_0]),
                ("user", [fr_part_0]),
                ("agent", [fc_part_1]),
                ("user", [fr_part_1]),
            ],
            [
                ("user", "Add one to 5"),
                ("model", fc_part_anon),
                ("user", fr_part_anon),
                ("model", fc_part_anon),
                ("user", fr_part_anon),
            ],
        ),
        (
            [
                ("user", [greeting_text_part]),
                ("agent", [fc_part_0]),
                ("user", [fr_part_0]),
                ("agent", [fc_part_1]),
                ("user", [fr_part_0, fr_part_1]),
            ],
            [
                ("user", "Add one to 5"),
                # ("model", fc_part_anon),
                # ("user", [fr_part_anon, fr_part_anon]),  # BUG!!!! This is too many responses!!!
                # ("model", fc_part_anon),
                # ("user", [fr_part_anon, fr_part_anon]),
                ("model", [fc_part_anon, fc_part_anon]),  # EXPECTED
                ("user", [fr_part_anon, fr_part_anon]),
            ],
        ),
        (
            [
                ("user", [greeting_text_part]),
                ("agent", [fc_part_0]),
                ("user", [fr_part_0]),
                ("user", [fr_part_alt_0]),
            ],
            [
                ("user", "Add one to 5"),
                ("model", fc_part_anon),
                ("user", fr_part_alt_anon),
            ],
        ),
        (
            [
                ("user", [greeting_text_part]),
                ("agent", [fc_part_0]),
                ("user", [fr_part_0]),
                ("agent", [fc_part_1]),
                ("user", [fr_part_1]),
                ("user", [fr_part_alt_0]),
            ],
            [
                ("user", greeting_text_part.text),
                ("model", [fc_part_anon, fc_part_anon]),
                ("user", [fr_part_anon, fr_part_alt_anon]),
            ],
        ),
        (
            [
                ("user", [greeting_text_part]),
                ("agent", [fc_part_0]),
                ("user", [fr_part_0]),
                ("agent", [fc_part_1]),
                ("user", [fr_part_1, fr_part_alt_0]),
            ],
            [
                ("user", "Add one to 5"),
                ("model", [fc_part_anon, fc_part_anon]),
                ("user", [fr_part_anon, fr_part_alt_anon]),
            ],
        ),
        (
            [
                ("user", [greeting_text_part]),
                ("agent", [fc_part_0]),
                ("user", [fr_part_0]),
                ("agent", [fc_part_1]),
                ("user", [fr_part_1, fr_part_alt_0]),
                ("agent", [agent_summary_text_part]),
                ("user", [continuation_text_part]),
            ],
            [
                ("user", greeting_text_part.text),
                ("model", [fc_part_anon, fc_part_anon]),
                ("user", [fr_part_anon, fr_part_alt_anon]),
                ("model", agent_summary_text_part.text),
                ("user", continuation_text_part.text),
            ],
        ),
    ],
)
async def test_prepare_contents(simple_events, simple_contents):
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=Agent(
          model="gemini-1.5-flash",
          name="agent",
      )
  )
  invocation_context.session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_id",
  )

  events = [
      Event(
          invocation_id=invocation_context.invocation_id,
          author=author,
          content=types.Content(role="user" if author == "user" else "model", parts=parts),
      )
      for author, parts in simple_events
  ]
  invocation_context.session.events.extend(events)

  async for _ in contents.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert simple_contents == simplify_contents(request.contents)
