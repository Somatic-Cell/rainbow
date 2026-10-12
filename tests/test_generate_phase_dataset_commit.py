"""Commit recovery tests. GPU output is a synthetic, explicitly mocked record.

Filesystem operations are real except where a named test injects a specific
failure. A real Windows sharing-lock test is skipped outside Windows.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import test_generate_phase_dataset as fixture

batch = fixture.batch


def win_error(code: int = 5) -> OSError:
    exc = PermissionError(errno.EACCES, "injected Windows sharing/access failure")
    exc.winerror = code
    return exc


class CommitTests(unittest.TestCase):
    setUp = fixture.BatchTests.setUp
    save = fixture.BatchTests.save
    invoke = fixture.BatchTests.invoke
    mock_generate = fixture.BatchTests.mock_generate
    run_some = fixture.BatchTests.run_some

    def paths(self):
        p = self.out/"records/i0000/w0000"
        return p, Path(str(p)+".part"), self.out/"receipts/i0000_w0000.json"

    def ledger(self):
        return self.out/"failures/i0000_w0000.json"

    def journal(self):
        return self.out/"commit_recoveries/i0000_w0000.json"

    def commit_failure(self, cmd, **kw):
        self.mock_generate(cmd, **kw)  # explicit synthetic CDF fixture, no GPU
        out = Path(cmd[cmd.index("--out")+1])
        stage = Path(str(out)+".part")
        os.rename(out, stage)
        kw["stdout"].write(
            f'rainbow_generate: rename: Access is denied.: "{stage}", "{out}"\n'.encode())
        return types.SimpleNamespace(returncode=1)

    def make_previous_failure(self):
        with patch.object(batch.subprocess, "run", side_effect=self.commit_failure):
            with self.assertRaises(RuntimeError):
                self.invoke("--execute", "--continue-on-error", "--limit", "1")
        self.assertTrue(self.paths()[1].exists())

    def repair_only(self):
        with patch.object(batch.subprocess, "run", side_effect=AssertionError("GPU must NOT be rerun")):
            return self.invoke("--execute", "--resume", "--recover-commits", "--retry-failed")

    def test_exact_windows_log_parsed_without_inferring_lock_owner(self):
        log = self.root/"windows.log"
        log.write_text('rainbow_generate: rename: Access is denied.: "C:\\data\\w0076.part", "C:\\data\\w0076"\n')
        value = batch.rename_failure(log, 1)
        self.assertEqual(value["source"], r"C:\data\w0076.part")
        self.assertEqual(value["os_message"], "Access is denied.")
        self.assertIsNone(batch.rename_failure(log, 0))
        self.assertIsNone(batch.rename_failure(log, 3221225477))
        self.assertEqual(batch.log_path_key(r"C:\data\w0076"), batch.log_path_key(r"c:/DATA/w0076"))

    def test_classifier_retains_actual_non_cdf_error(self):
        log = self.root/"error.log"
        line = 'rainbow_generate: rename: Access is denied.: "a.part", "a"'
        log.write_text(line+"\n")
        detail = batch.inspect_job_failure(log, 1)
        self.assertEqual(detail["message"], line)
        self.assertEqual(detail["generator_error_lines"], [line])
        self.assertFalse(detail["continuable"])  # recovery is NOT blind continuation

    def test_recovery_not_enabled_by_default(self):
        self.make_previous_failure()
        with self.assertRaises(ValueError):
            self.invoke("--execute", "--resume", "--continue-on-error")
        self.assertFalse(self.paths()[0].exists())

    def test_new_failures_recovered_and_queue_continues(self):
        with patch.object(batch.subprocess, "run", side_effect=self.commit_failure):
            self.assertEqual(self.invoke("--execute", "--recover-commits", "--continue-on-error"), 0)
        self.assertEqual(len(self.calls), 4)  # one invocation per condition, not two
        state = batch.load_json(self.out/"progress.json")
        self.assertTrue(state["complete"])
        self.assertEqual(state["failed"], 0)
        self.assertEqual(len(list((self.out/"commit_recoveries").glob("*.json"))), 4)
        self.assertEqual(batch.load_json(self.ledger())["attempts"][0]["returncode"], 1)

    def test_old_generic_failure_ledger_can_be_recovered(self):
        self.make_previous_failure()
        value = batch.load_json(self.ledger())
        value["runner_version"] = "rainbow.condition_batch.failure_queue.v1"
        value["attempts"][0]["message"] = "Nonzero generator exit without a recognized safe-to-defer rejection."
        value["attempts"][0].pop("generator_error_lines", None)
        batch.write_json(self.ledger(), value)
        before = batch.record_hashes(self.paths()[1])
        self.assertEqual(self.repair_only(), 0)
        self.assertEqual(batch.record_hashes(self.paths()[0]), before)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.paths()[2].exists())
        self.assertFalse(self.paths()[1].exists())
        self.assertEqual(batch.load_json(self.ledger())["state"], "resolved")
        self.assertEqual(batch.load_json(self.journal())["state"], "committed")

    def test_recovery_then_one_new_condition(self):
        self.make_previous_failure()
        with patch.object(batch.subprocess, "run", side_effect=self.mock_generate):
            self.assertEqual(self.invoke("--execute", "--resume", "--recover-commits", "--limit", "1"), 0)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(batch.load_json(self.out/"progress.json")["completed"], 2)

    def test_resume_respects_binary_hash(self):
        self.make_previous_failure()
        self.exe.write_bytes(b"changed")
        with self.assertRaises(ValueError): self.repair_only()
        self.assertTrue(self.paths()[1].exists())
        self.assertFalse(self.paths()[2].exists())

    def test_resume_respects_plan_settings(self):
        self.make_previous_failure()
        self.raw["temperature_c"] = 21
        self.save()
        with self.assertRaises(ValueError): self.repair_only()

    def test_failure_signature_mismatch_rejected(self):
        self.make_previous_failure()
        value = batch.load_json(self.ledger()); value["signature"] = "different"
        batch.write_json(self.ledger(), value)
        with self.assertRaises(ValueError): self.repair_only()

    def test_wrong_log_paths_not_adopted(self):
        self.make_previous_failure()
        log = self.out/"logs/i0000_w0000.log"
        log.write_text(log.read_text().replace("w0000", "w0001"))
        with self.assertRaises(ValueError): self.repair_only()
        self.assertFalse(self.paths()[2].exists())

    def test_no_failure_log_no_adoption(self):
        self.make_previous_failure()
        self.ledger().unlink()
        with self.assertRaises(ValueError): self.repair_only()

    def test_log_path_escape_rejected(self):
        self.make_previous_failure()
        value = batch.load_json(self.ledger()); value["attempts"][0]["log"] = "../outside.log"
        batch.write_json(self.ledger(), value)
        with self.assertRaises(ValueError): self.repair_only()

    def test_numerical_rejection_is_not_commit_recovery(self):
        def action(cmd, **kw):
            self.calls.append(cmd)
            kw["stdout"].write(fixture.CDF_REJECTION.encode())
            return types.SimpleNamespace(returncode=1)
        with patch.object(batch.subprocess, "run", side_effect=action):
            self.assertEqual(self.invoke("--execute", "--recover-commits", "--continue-on-error", "--limit", "1"), 2)
        self.assertFalse((self.out/"commit_recoveries").exists())

    def test_truncated_payload_rejected(self):
        self.make_previous_failure()
        p = self.paths()[1]/"theta_given_phi_cdf.npy"
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(ValueError): self.repair_only()
        self.assertTrue(p.exists()); self.assertFalse(self.paths()[2].exists())

    def test_same_size_nan_payload_rejected(self):
        self.make_previous_failure()
        p = self.paths()[1]/"theta_given_phi_cdf.npy"
        fixture.write_npy(p, (4,3), [0.0,float("nan"),1.0]*4)
        with self.assertRaises(ValueError): self.repair_only()

    def test_decreasing_cdf_rejected(self):
        self.make_previous_failure()
        fixture.write_npy(self.paths()[1]/"phi_cdf.npy", (5,), [0.0,.5,.4,.8,1.0])
        with self.assertRaises(ValueError): self.repair_only()

    def test_nonunit_cdf_endpoint_rejected(self):
        self.make_previous_failure()
        fixture.write_npy(self.paths()[1]/"phi_cdf.npy", (5,), [0.0,.25,.5,.75,.99999])
        with self.assertRaises(ValueError): self.repair_only()

    def test_repeated_edge_rejected(self):
        self.make_previous_failure()
        fixture.write_npy(self.paths()[1]/"u_edges.npy", (3,), [0.0,0.0,1.0])
        with self.assertRaises(ValueError): self.repair_only()

    def test_nonuniform_cdf_exact_mean_cosine(self):
        self.make_previous_failure()
        part = self.paths()[1]
        fixture.write_npy(part/"u_edges.npy", (3,), [0.0, .5, 1.0])
        fixture.write_npy(part/"theta_given_phi_cdf.npy", (4,3), [0.0, .75, 1.0]*4)
        meta = batch.load_json(part/"metadata.json"); meta["hg"]["g"] = .25
        batch.write_json(part/"metadata.json", meta)
        self.assertEqual(self.repair_only(), 0)
        validation = batch.load_json(self.journal())["cdf_validation"]
        self.assertAlmostEqual(validation["mass"], 1.0)
        self.assertAlmostEqual(validation["g"], .25)

    def test_saved_g_mismatch_rejected(self):
        self.make_previous_failure()
        p=self.paths()[1]/"metadata.json"; v=batch.load_json(p); v["hg"]["g"] = .01
        batch.write_json(p, v)
        with self.assertRaises(ValueError): self.repair_only()

    def test_metadata_optical_failure_not_accepted(self):
        self.make_previous_failure()
        p=self.paths()[1]/"metadata.json"; v=batch.load_json(p); v["quality"]["incomplete_directions"] = 1
        batch.write_json(p, v)
        with self.assertRaises(ValueError): self.repair_only()

    def test_unknown_extra_file_not_deleted(self):
        self.make_previous_failure()
        extra=self.paths()[1]/"keep.txt"; extra.write_text("preserve")
        with self.assertRaises(ValueError): self.repair_only()
        self.assertEqual(extra.read_text(), "preserve")

    def test_existing_destination_not_replaced(self):
        self.make_previous_failure()
        self.paths()[0].mkdir(); p=self.paths()[0]/"keep"; p.write_text("preserve")
        with self.assertRaises(ValueError): self.repair_only()
        self.assertEqual(p.read_text(), "preserve")
        self.assertTrue(self.paths()[1].exists())

    def test_manual_rename_without_journal_does_not_forge_receipt(self):
        self.make_previous_failure()
        os.rename(self.paths()[1], self.paths()[0])
        with self.assertRaises(ValueError): self.repair_only()
        self.assertFalse(self.paths()[2].exists())

    def test_permission_retry_then_success(self):
        self.make_previous_failure()
        original = Path.rename
        count=0
        def rename(p, target):
            nonlocal count
            count += 1
            if count < 3: raise win_error(5 if count==1 else 32)
            return original(p, target)
        with patch.object(Path, "rename", rename), patch.object(batch.time, "sleep") as sleep:
            self.assertEqual(self.repair_only(), 0)
        self.assertEqual(count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(batch.load_json(self.journal())["rename_attempts"], 3)

    def test_permanent_denial_bounded_and_preserved(self):
        self.make_previous_failure()
        before = batch.record_hashes(self.paths()[1])
        with patch.object(Path, "rename", side_effect=win_error(5)) as rename, patch.object(batch.time, "sleep") as sleep:
            with self.assertRaises(PermissionError): self.repair_only()
        self.assertEqual(rename.call_count, 7)
        self.assertEqual(sleep.call_count, 6)
        self.assertEqual(batch.record_hashes(self.paths()[1]), before)
        self.assertFalse(self.paths()[2].exists())
        self.assertEqual(batch.load_json(self.journal())["state"], "validated")
        self.assertEqual(self.repair_only(), 0)  # lock gone; reuse the same bytes

    def test_non_permission_io_error_not_retried(self):
        self.make_previous_failure()
        with patch.object(Path,"rename", side_effect=OSError(errno.EIO,"injected disk I/O failure")) as rename, patch.object(batch.time,"sleep") as sleep:
            with self.assertRaises(OSError): self.repair_only()
        self.assertEqual(rename.call_count,1); sleep.assert_not_called()
        self.assertTrue(self.paths()[1].exists())

    def test_crash_after_rename_before_receipt_recovered_without_gpu(self):
        self.make_previous_failure()
        original = batch.write_json
        def write(path, value):
            if path.parent.name == "receipts": raise OSError("injected failure before receipt")
            return original(path, value)
        with patch.object(batch, "write_json", write):
            with self.assertRaises(OSError): self.repair_only()
        self.assertTrue(self.paths()[0].exists()); self.assertFalse(self.paths()[1].exists())
        self.assertFalse(self.paths()[2].exists()); self.assertTrue(self.journal().exists())
        self.assertEqual(self.repair_only(),0)
        self.assertEqual(len(self.calls),1)

    def test_journal_rejects_modified_payload(self):
        self.make_previous_failure()
        with patch.object(Path,"rename",side_effect=win_error()), patch.object(batch.time,"sleep"):
            with self.assertRaises(PermissionError): self.repair_only()
        # Remains a valid CDF with g=0, but it is not the validated transaction.
        fixture.write_npy(self.paths()[1]/"phi_cdf.npy",(5,),[0.0,.1,.5,.9,1.0])
        with self.assertRaises(ValueError): self.repair_only()
        self.assertFalse(self.paths()[2].exists())

    def test_journal_rejects_modified_original_log(self):
        self.make_previous_failure()
        with patch.object(Path,"rename",side_effect=win_error()), patch.object(batch.time,"sleep"):
            with self.assertRaises(PermissionError): self.repair_only()
        log=self.out/"logs/i0000_w0000.log"; log.write_text("changed preamble\n"+log.read_text())
        with self.assertRaises(ValueError): self.repair_only()

    def test_symlink_payload_refused(self):
        self.make_previous_failure()
        p=self.paths()[1]/"u_edges.npy"; copy=self.root/"copy.npy"
        copy.write_bytes(p.read_bytes()); p.unlink()
        try: p.symlink_to(copy)
        except (OSError,NotImplementedError): self.skipTest("symlink unavailable")
        with self.assertRaises(ValueError): self.repair_only()
        self.assertTrue(copy.exists()); self.assertFalse(self.paths()[2].exists())

    def test_dry_run_recovery_does_not_touch_existing_files(self):
        self.make_previous_failure()
        before=batch.record_hashes(self.paths()[1])
        self.assertEqual(self.invoke("--resume","--recover-commits","--dry-run"),0)
        self.assertEqual(before,batch.record_hashes(self.paths()[1])); self.assertFalse(self.journal().exists())

    def test_nominal_success_never_full_audited_by_recovery(self):
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate), patch.object(batch,"audit_saved_cdf",side_effect=AssertionError("no added full reads on normal success")):
            self.assertEqual(self.invoke("--execute","--recover-commits"),0)


@unittest.skipUnless(os.name == "nt", "real Windows directory-share-lock test")
class WindowsLockTest(unittest.TestCase):
    def test_child_file_handle_released_before_retry(self):
        import ctypes
        from ctypes import wintypes
        import threading
        import time
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.CreateFileW.argtypes=(wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE)
        kernel.CreateFileW.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=(wintypes.HANDLE,); kernel.CloseHandle.restype=wintypes.BOOL
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/"record.part"; source.mkdir(); (source/"data").write_bytes(b"abc")
            handle=kernel.CreateFileW(str(source/"data"),0x80000000,0x1|0x2,None,3,0x80,None)
            if handle == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
            def release():
                time.sleep(.25)
                kernel.CloseHandle(handle)
            thread=threading.Thread(target=release); thread.start()
            try:
                info=batch.rename_with_retry(source,Path(temp)/"record")
            finally: thread.join()
            self.assertGreaterEqual(info["rename_attempts"],2)
            self.assertEqual((Path(temp)/"record/data").read_bytes(),b"abc")


if __name__ == "__main__":
    unittest.main()
