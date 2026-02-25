import json
import os
import unittest

os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import _GlobalState
from sglang.srt.managers.io_struct import (
    LoRAUpdateOutput,
    UpdateLoRAAdapterFromTensorsReqInput,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")


class _TokenizerManagerSuccess:
    async def update_lora_adapter_from_tensors(self, obj, request):
        return LoRAUpdateOutput(success=True, loaded_adapters={obj.lora_name: None})


class _TokenizerManagerFailure:
    async def update_lora_adapter_from_tensors(self, obj, request):
        return LoRAUpdateOutput(success=False, error_message="update failed")


class TestUpdateLoRAAdapterFromTensorsEndpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._prev_state = http_server.get_global_state()

    def tearDown(self):
        http_server.set_global_state(self._prev_state)

    async def test_update_lora_adapter_from_tensors_success(self):
        http_server.set_global_state(
            _GlobalState(
                tokenizer_manager=_TokenizerManagerSuccess(),
                template_manager=None,
                scheduler_info={},
            )
        )
        req = UpdateLoRAAdapterFromTensorsReqInput(
            lora_name="adapter",
            config_dict={},
            serialized_tensors="serialized",
        )

        response = await http_server.update_lora_adapter_from_tensors(req, None)
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertIn("adapter", payload["loaded_adapters"])

    async def test_update_lora_adapter_from_tensors_failure(self):
        http_server.set_global_state(
            _GlobalState(
                tokenizer_manager=_TokenizerManagerFailure(),
                template_manager=None,
                scheduler_info={},
            )
        )
        req = UpdateLoRAAdapterFromTensorsReqInput(
            lora_name="adapter",
            config_dict={},
            serialized_tensors="serialized",
        )

        response = await http_server.update_lora_adapter_from_tensors(req, None)
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 400)
        self.assertFalse(payload["success"])
        self.assertIn("update failed", payload["error_message"])

if __name__ == "__main__":
    unittest.main()
