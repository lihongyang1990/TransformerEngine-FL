# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

class TestBackendImplKind(unittest.TestCase):
    def test_kind_values(self):
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind

        self.assertEqual(BackendImplKind.DEFAULT.value, "default")
        self.assertEqual(BackendImplKind.REFERENCE.value, "reference")
        self.assertEqual(BackendImplKind.VENDOR.value, "vendor")

    def test_kind_is_string_enum(self):
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind

        self.assertIsInstance(BackendImplKind.DEFAULT, str)
        self.assertEqual(str(BackendImplKind.DEFAULT), "default")


class TestOpImpl(unittest.TestCase):
    def test_opimpl_creation(self):
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl

        def dummy_fn():
            pass

        impl = OpImpl(
            op_name="test_op",
            impl_id="vendor.test.v1",
            kind=BackendImplKind.VENDOR,
            fn=dummy_fn,
            vendor="test",
            priority=100,
        )

        self.assertEqual(impl.op_name, "test_op")
        self.assertEqual(impl.impl_id, "vendor.test.v1")
        self.assertEqual(impl.kind, BackendImplKind.VENDOR)
        self.assertEqual(impl.vendor, "test")
        self.assertEqual(impl.priority, 100)

    def test_opimpl_vendor_required_for_vendor_kind(self):
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl

        def dummy_fn():
            pass

        with self.assertRaises(ValueError):
            OpImpl(
                op_name="test_op",
                impl_id="vendor.test.v1",
                kind=BackendImplKind.VENDOR,
                fn=dummy_fn,
                vendor=None,
            )

    def test_opimpl_is_available(self):
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl

        def available_fn():
            pass
        available_fn._is_available = lambda: True

        def unavailable_fn():
            pass
        unavailable_fn._is_available = lambda: False

        impl1 = OpImpl(
            op_name="test_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=available_fn,
        )
        self.assertTrue(impl1.is_available())

        impl2 = OpImpl(
            op_name="test_op",
            impl_id="default.test2",
            kind=BackendImplKind.DEFAULT,
            fn=unavailable_fn,
        )
        self.assertFalse(impl2.is_available())


class TestMatchToken(unittest.TestCase):
    def test_match_default(self):
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl, match_token

        def dummy():
            pass

        impl = OpImpl(
            op_name="test",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=dummy,
        )

        self.assertTrue(match_token(impl, "default"))
        self.assertFalse(match_token(impl, "reference"))
        self.assertFalse(match_token(impl, "vendor"))

    def test_match_vendor_with_name(self):
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl, match_token

        def dummy():
            pass

        impl = OpImpl(
            op_name="test",
            impl_id="vendor.cuda.v1",
            kind=BackendImplKind.VENDOR,
            fn=dummy,
            vendor="cuda",
        )

        self.assertTrue(match_token(impl, "vendor"))
        self.assertTrue(match_token(impl, "vendor:cuda"))
        self.assertFalse(match_token(impl, "vendor:amd"))

    def test_match_impl_id(self):
        from transformer_engine.plugins.transformer_engine_fl.types import BackendImplKind, OpImpl, match_token

        def dummy():
            pass

        impl = OpImpl(
            op_name="test",
            impl_id="vendor.cuda.v1",
            kind=BackendImplKind.VENDOR,
            fn=dummy,
            vendor="cuda",
        )

        self.assertTrue(match_token(impl, "impl:vendor.cuda.v1"))
        self.assertFalse(match_token(impl, "impl:vendor.amd.v1"))


class TestSelectionPolicy(unittest.TestCase):
    def test_policy_defaults(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy

        policy = SelectionPolicy()

        self.assertTrue(policy.prefer_vendor)
        self.assertFalse(policy.strict)
        self.assertEqual(policy.per_op_order, ())
        self.assertEqual(policy.deny_vendors, frozenset())
        self.assertIsNone(policy.allow_vendors)

    def test_policy_from_dict(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy

        policy = SelectionPolicy.from_dict(
            prefer_vendor=False,
            strict=True,
            per_op_order={"rmsnorm_fwd": ["vendor:cuda", "default"]},
            deny_vendors={"amd"},
            allow_vendors={"cuda", "intel"},
        )

        self.assertFalse(policy.prefer_vendor)
        self.assertTrue(policy.strict)
        self.assertEqual(policy.get_per_op_order("rmsnorm_fwd"), ["vendor:cuda", "default"])
        self.assertIn("amd", policy.deny_vendors)
        self.assertIn("cuda", policy.allow_vendors)

    def test_policy_fingerprint(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy

        policy1 = SelectionPolicy()
        policy2 = SelectionPolicy(prefer_vendor=False)
        policy3 = SelectionPolicy()

        self.assertEqual(policy1.fingerprint(), policy3.fingerprint())

        self.assertNotEqual(policy1.fingerprint(), policy2.fingerprint())

    def test_policy_is_vendor_allowed(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy

        policy1 = SelectionPolicy.from_dict(deny_vendors={"amd"})
        self.assertTrue(policy1.is_vendor_allowed("cuda"))
        self.assertFalse(policy1.is_vendor_allowed("amd"))

        policy2 = SelectionPolicy.from_dict(allow_vendors={"cuda"})
        self.assertTrue(policy2.is_vendor_allowed("cuda"))
        self.assertFalse(policy2.is_vendor_allowed("amd"))

    def test_policy_get_default_order(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy

        policy1 = SelectionPolicy(prefer_vendor=True)
        self.assertEqual(policy1.get_default_order(), ["vendor", "default", "reference"])

        policy2 = SelectionPolicy(prefer_vendor=False)
        self.assertEqual(policy2.get_default_order(), ["default", "vendor", "reference"])

    def test_policy_immutable(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy

        policy = SelectionPolicy()

        with self.assertRaises(AttributeError):
            policy.prefer_vendor = False


class TestPolicyContext(unittest.TestCase):
    def test_policy_context_override(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy, get_policy, policy_context, reset_global_policy

        reset_global_policy()

        default_policy = get_policy()
        self.assertTrue(default_policy.prefer_vendor)

        override_policy = SelectionPolicy(prefer_vendor=False, strict=True)
        with policy_context(override_policy):
            ctx_policy = get_policy()
            self.assertFalse(ctx_policy.prefer_vendor)
            self.assertTrue(ctx_policy.strict)

        after_policy = get_policy()
        self.assertTrue(after_policy.prefer_vendor)

    def test_policy_context_nested(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy, get_policy, policy_context, reset_global_policy

        reset_global_policy()

        policy1 = SelectionPolicy(prefer_vendor=False)
        policy2 = SelectionPolicy(strict=True)

        with policy_context(policy1):
            self.assertFalse(get_policy().prefer_vendor)
            self.assertFalse(get_policy().strict)

            with policy_context(policy2):
                self.assertTrue(get_policy().strict)

            self.assertFalse(get_policy().strict)

    def test_policy_context_thread_isolation(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import SelectionPolicy, get_policy, policy_context, reset_global_policy

        reset_global_policy()

        results: Dict[str, bool] = {}

        def thread_with_policy():
            with policy_context(SelectionPolicy(prefer_vendor=False)):
                time.sleep(0.1)
                results["thread1"] = get_policy().prefer_vendor

        def thread_without_policy():
            time.sleep(0.05)
            results["thread2"] = get_policy().prefer_vendor

        t1 = threading.Thread(target=thread_with_policy)
        t2 = threading.Thread(target=thread_without_policy)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertFalse(results["thread1"])
        self.assertTrue(results["thread2"])


class TestPolicyEpoch(unittest.TestCase):
    def test_epoch_increments(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import get_policy_epoch, bump_policy_epoch

        epoch1 = get_policy_epoch()
        bump_policy_epoch()
        epoch2 = get_policy_epoch()

        self.assertEqual(epoch2, epoch1 + 1)

    def test_epoch_increments_on_policy_change(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import (
            SelectionPolicy,
            get_policy_epoch,
            set_global_policy,
            reset_global_policy,
        )

        reset_global_policy()
        epoch1 = get_policy_epoch()

        set_global_policy(SelectionPolicy(prefer_vendor=False))
        epoch2 = get_policy_epoch()

        self.assertGreater(epoch2, epoch1)


class TestRegistryThreadSafety(unittest.TestCase):
    def test_concurrent_backend_access(self):
        from transformer_engine.plugins.transformer_engine_fl.registry import list_backends, get_registered_backend_names

        errors: List[Exception] = []

        def worker():
            try:
                for _ in range(100):
                    list_backends()
                    get_registered_backend_names()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors during concurrent access: {errors}")


class TestConvenienceFunctions(unittest.TestCase):
    def test_with_strict_mode(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import get_policy, with_strict_mode, reset_global_policy

        reset_global_policy()

        self.assertFalse(get_policy().strict)

        with with_strict_mode():
            self.assertTrue(get_policy().strict)

        self.assertFalse(get_policy().strict)

    def test_with_vendor_preference(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import get_policy, with_vendor_preference, reset_global_policy

        reset_global_policy()

        self.assertTrue(get_policy().prefer_vendor)

        with with_vendor_preference(False):
            self.assertFalse(get_policy().prefer_vendor)

        self.assertTrue(get_policy().prefer_vendor)

    def test_with_denied_vendors(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import get_policy, with_denied_vendors, reset_global_policy

        reset_global_policy()

        self.assertEqual(get_policy().deny_vendors, frozenset())

        with with_denied_vendors("amd", "intel"):
            policy = get_policy()
            self.assertIn("amd", policy.deny_vendors)
            self.assertIn("intel", policy.deny_vendors)

        self.assertEqual(get_policy().deny_vendors, frozenset())


class TestDiscovery(unittest.TestCase):
    def test_discovered_plugins_list(self):
        from transformer_engine.plugins.transformer_engine_fl.discovery import get_discovered_plugins, clear_discovered_plugins

        clear_discovered_plugins()
        plugins = get_discovered_plugins()

        self.assertIsInstance(plugins, list)

    def test_plugin_group_constant(self):
        from transformer_engine.plugins.transformer_engine_fl.discovery import PLUGIN_GROUP

        self.assertEqual(PLUGIN_GROUP, "te_fl.plugins")

    def test_plugin_modules_env_constant(self):
        from transformer_engine.plugins.transformer_engine_fl.discovery import PLUGIN_MODULES_ENV

        self.assertEqual(PLUGIN_MODULES_ENV, "TE_FL_PLUGIN_MODULES")


class TestEnvironmentVariables(unittest.TestCase):
    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_policy_from_env_prefer_vendor(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import policy_from_env

        os.environ["TE_FL_PREFER_VENDOR"] = "0"
        policy = policy_from_env()
        self.assertFalse(policy.prefer_vendor)

        os.environ["TE_FL_PREFER_VENDOR"] = "1"
        policy = policy_from_env()
        self.assertTrue(policy.prefer_vendor)

    def test_policy_from_env_strict(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import policy_from_env

        os.environ["TE_FL_STRICT"] = "1"
        policy = policy_from_env()
        self.assertTrue(policy.strict)

        os.environ["TE_FL_STRICT"] = "0"
        policy = policy_from_env()
        self.assertFalse(policy.strict)

    def test_policy_from_env_deny_vendors(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import policy_from_env

        os.environ["TE_FL_DENY_VENDORS"] = "amd,intel"
        policy = policy_from_env()
        self.assertIn("amd", policy.deny_vendors)
        self.assertIn("intel", policy.deny_vendors)

    def test_policy_from_env_allow_vendors(self):
        from transformer_engine.plugins.transformer_engine_fl.policy import policy_from_env

        os.environ["TE_FL_ALLOW_VENDORS"] = "cuda"
        policy = policy_from_env()
        self.assertIsNotNone(policy.allow_vendors)
        self.assertIn("cuda", policy.allow_vendors)


class TestCacheInvalidation(unittest.TestCase):
    def test_clear_cache(self):
        from transformer_engine.plugins.transformer_engine_fl.operator_registry import clear_cache, get_cache_stats

        clear_cache()
        stats = get_cache_stats()
        self.assertEqual(stats["size"], 0)

    def test_cache_stats(self):
        from transformer_engine.plugins.transformer_engine_fl.operator_registry import get_cache_stats

        stats = get_cache_stats()
        self.assertIn("size", stats)
        self.assertIn("local_epoch", stats)


class TestIntegration(unittest.TestCase):
    def test_full_import(self):
        from transformer_engine.plugins.transformer_engine_fl import (
            BackendImplKind,
            OpImpl,
            TEXBackendBase,
            TEXModule,
            register_backend,
            get_backend,
            list_backends,
            SelectionPolicy,
            get_policy,
            policy_context,
            with_fallback,
            with_debug,
            set_operator_backend,
            resolve_operator_impl,
            discover_plugins,
            PLUGIN_GROUP,
        )

        self.assertIsNotNone(BackendImplKind)
        self.assertIsNotNone(SelectionPolicy)

    def test_list_backends_returns_data(self):
        from transformer_engine.plugins.transformer_engine_fl import list_backends

        backends = list_backends()
        self.assertIsInstance(backends, list)

        if backends:
            backend = backends[0]
            self.assertIn("name", backend)
            self.assertIn("priority", backend)
            self.assertIn("available", backend)


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestBackendImplKind))
    suite.addTests(loader.loadTestsFromTestCase(TestOpImpl))
    suite.addTests(loader.loadTestsFromTestCase(TestMatchToken))
    suite.addTests(loader.loadTestsFromTestCase(TestSelectionPolicy))
    suite.addTests(loader.loadTestsFromTestCase(TestPolicyContext))
    suite.addTests(loader.loadTestsFromTestCase(TestPolicyEpoch))
    suite.addTests(loader.loadTestsFromTestCase(TestRegistryThreadSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestDiscovery))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentVariables))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheInvalidation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
