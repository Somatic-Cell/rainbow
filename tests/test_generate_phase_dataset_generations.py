"""Real temporary files + mocked optical child process; NOT GPU validation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import types
import unittest
from unittest.mock import patch

import test_generate_phase_dataset as fixture
import test_generate_phase_dataset_commit as commit_fixture

batch = fixture.batch
gen = batch.generations


class GenerationTests(unittest.TestCase):
    setUp = fixture.BatchTests.setUp
    save = fixture.BatchTests.save
    invoke = fixture.BatchTests.invoke
    mock_generate = fixture.BatchTests.mock_generate
    run_some = fixture.BatchTests.run_some
    fail_job = fixture.FailureTests.fail_job
    commit_failure = commit_fixture.CommitTests.commit_failure

    def capture(self):
        return {p.relative_to(self.out).as_posix(): p.read_bytes()
                for p in self.out.rglob("*") if p.is_file()}

    def change_binary(self, suffix="new"):
        backup = self.root / ("backup_"+suffix)
        backup.mkdir()
        shutil.copyfile(self.exe, backup/self.exe.name)
        shutil.copytree(self.root/"modules", backup/"modules")
        self.exe.write_bytes(self.exe.read_bytes()+suffix.encode())
        module = self.root/"modules/patch_build.fatbin"
        module.write_bytes(module.read_bytes()+suffix.encode())
        return backup/self.exe.name

    def register_args(self, backup):
        return ["--register-generation", "regular64 tested change",
                "--previous-generator", str(backup)]

    def history(self):
        return json.loads((self.out/"generation_history.json").read_text())

    def make_fail_one(self):
        with patch.object(batch.subprocess, "run", side_effect=self.fail_job):
            self.invoke("--execute", "--continue-on-error", "--limit", "1")

    def test_plain_resume_after_changed_binary_rejected(self):
        self.run_some(1)
        self.change_binary()
        before = self.capture()
        with self.assertRaisesRegex(ValueError,"binaries changed"):
            self.invoke("--execute", "--resume")
        self.assertEqual(before,self.capture())

    def test_register_keeps_original_plan_receipt_and_array_bytes(self):
        self.run_some(1)
        before = self.capture()
        old = json.loads(before["batch_plan.json"])["signature"]
        backup = self.change_binary()
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.assertEqual(self.invoke("--execute","--resume",*self.register_args(backup)),0)
        for name, value in before.items():
            if name.startswith(("records/","receipts/")) or name=="batch_plan.json":
                self.assertEqual((self.out/name).read_bytes(),value,name)
        history=self.history();new=history["active_signature"]
        self.assertNotEqual(old,new)
        self.assertEqual(len(history["generations"]),2)
        self.assertEqual(json.loads((self.out/"receipts/i0000_w0000.json").read_text())["signature"],old)
        self.assertEqual(json.loads((self.out/"receipts/i0000_w0001.json").read_text())["signature"],new)
        with patch.object(batch.subprocess,"run",side_effect=AssertionError("Already done")):
            self.assertEqual(self.invoke("--execute","--resume"),0)

    def test_same_binary_registration_is_idempotent(self):
        self.run_some(1);backup=self.change_binary()
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.invoke("--execute","--resume",*self.register_args(backup),"--limit","1")
        original=(self.out/"generation_history.json").read_bytes()
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.invoke("--execute","--resume",*self.register_args(backup))
        self.assertEqual((self.out/"generation_history.json").read_bytes(),original)

    def test_conditions_change_rejected_even_with_authorization(self):
        self.run_some(1);backup=self.change_binary()
        self.raw["solver"]["grid"]=11;self.save()
        with self.assertRaisesRegex(ValueError,"conditions"):
            self.invoke("--execute","--resume",*self.register_args(backup))
        self.assertFalse((self.out/"generation_history.json").exists())

    def test_wrong_previous_backup_rejected(self):
        self.run_some(1);backup=self.change_binary()
        backup.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError,"previous binaries"):
            self.invoke("--execute","--resume",*self.register_args(backup))
        self.assertFalse((self.out/"generation_history.json").exists())

    def test_previous_backup_missing_module_rejected(self):
        self.run_some(1);backup=self.change_binary()
        (backup.parent/"modules/phase_cdf.fatbin").unlink()
        with self.assertRaises(FileNotFoundError):
            self.invoke("--execute","--resume",*self.register_args(backup))

    def test_register_without_backup_or_resume_rejected(self):
        self.run_some(1);self.change_binary()
        with self.assertRaises(ValueError):
            self.invoke("--execute","--resume","--register-generation","fix")
        with self.assertRaises(ValueError):
            self.invoke("--execute","--register-generation","fix")

    def test_registry_plan_hash_tampering_rejected(self):
        self.run_some(1);backup=self.change_binary()
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.invoke("--execute","--resume",*self.register_args(backup),"--limit","1")
        h=self.history();h["generations"][1]["plan"]["binary_sha256"]["generator"]="0"*64
        (self.out/"generation_history.json").write_text(json.dumps(h))
        with self.assertRaisesRegex(ValueError,"signature"):
            self.invoke("--execute","--resume")

    def test_unknown_receipt_generation_rejected(self):
        self.run_some(1)
        path=self.out/"receipts/i0000_w0000.json"
        r=json.loads(path.read_text());r["signature"]="0"*64;path.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,"unregistered generation"):
            self.invoke("--execute","--resume")

    def test_retry_failed_and_pending_selects_both(self):
        self.make_fail_one();backup=self.change_binary()
        old_failure=(self.out/"failures/i0000_w0000.json").read_bytes()
        old_attempts=json.loads(old_failure)["attempts"]
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate) as child:
            self.assertEqual(self.invoke("--execute","--resume","--retry-failed-and-pending",
                                        *self.register_args(backup)),0)
            self.assertEqual(child.call_count,4)
        f=json.loads((self.out/"failures/i0000_w0000.json").read_text())
        self.assertEqual(f["state"],"resolved")
        self.assertEqual(f["attempts"],old_attempts)
        self.assertEqual(f["resolved_generation_signature"],self.history()["active_signature"])

    def test_new_failure_attempt_keeps_old_generation_identity(self):
        self.make_fail_one();backup=self.change_binary()
        file=self.out/"failures/i0000_w0000.json"
        old=json.loads(file.read_text())
        with patch.object(batch.subprocess,"run",side_effect=self.fail_job):
            self.assertEqual(self.invoke("--execute","--resume","--retry-failed-and-pending",
                              "--continue-on-error","--limit","1",*self.register_args(backup)),2)
        f=json.loads(file.read_text())
        self.assertEqual(f["signature"],old["signature"])
        self.assertEqual(f["attempts"][0],old["attempts"][0])
        self.assertEqual(f["attempts"][1]["generation_signature"],self.history()["active_signature"])
        # Skip old/new failures on ordinary continued resume, without redoing it.
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate) as child:
            self.assertEqual(self.invoke("--execute","--resume","--continue-on-error"),2)
            self.assertEqual(child.call_count,3)

    def test_legacy_attempt_without_generation_tag_is_preserved(self):
        self.make_fail_one()
        p=self.out/"failures/i0000_w0000.json";f=json.loads(p.read_text())
        f["attempts"][0].pop("generation_signature");p.write_text(json.dumps(f))
        backup=self.change_binary()
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.assertEqual(self.invoke("--execute","--resume","--retry-failed-and-pending",
                                        *self.register_args(backup)),0)
        self.assertNotIn("generation_signature",json.loads(p.read_text())["attempts"][0])

    def test_old_commit_failure_recovered_under_its_original_generation(self):
        with patch.object(batch.subprocess,"run",side_effect=self.commit_failure):
            with self.assertRaises(RuntimeError):
                self.invoke("--execute","--limit","1")
        old=json.loads((self.out/"batch_plan.json").read_text())["signature"]
        backup=self.change_binary()
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate) as child:
            self.assertEqual(self.invoke("--execute","--resume","--recover-commits",
                                        *self.register_args(backup)),0)
            self.assertEqual(child.call_count,3)
        receipt=json.loads((self.out/"receipts/i0000_w0000.json").read_text())
        self.assertEqual(receipt["signature"],old)

    def test_new_commit_failure_after_old_numerical_failure(self):
        self.make_fail_one();backup=self.change_binary()
        with patch.object(batch.subprocess,"run",side_effect=self.commit_failure):
            self.assertEqual(self.invoke("--execute","--resume","--recover-commits",
                            "--retry-failed-and-pending",*self.register_args(backup)),0)
        f=json.loads((self.out/"failures/i0000_w0000.json").read_text())
        self.assertEqual(len(f["attempts"]),2)
        self.assertEqual(f["state"],"resolved")
        self.assertEqual(json.loads((self.out/"receipts/i0000_w0000.json").read_text())["signature"],
                         self.history()["active_signature"])

    def test_rollback_rejected(self):
        self.run_some(1);backup=self.change_binary()
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.invoke("--execute","--resume",*self.register_args(backup))
        shutil.copyfile(backup,self.exe)
        for name in batch.MODULES:shutil.copyfile(backup.parent/"modules"/name,self.root/"modules"/name)
        with self.assertRaisesRegex(ValueError,"rollback"):
            self.invoke("--execute","--resume")

    def test_third_generation_requires_backup_of_second(self):
        self.run_some(1);first_backup=self.change_binary()
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.invoke("--execute","--resume",*self.register_args(first_backup),"--limit","1")
        second_backup=self.change_binary("third")
        with self.assertRaises(ValueError):
            self.invoke("--execute","--resume",*self.register_args(first_backup))
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.invoke("--execute","--resume",*self.register_args(second_backup))
        self.assertEqual(len(self.history()["generations"]),3)

    def test_dry_run_does_not_register_or_retire(self):
        self.run_some(1);backup=self.change_binary();before=self.capture()
        self.invoke("--dry-run","--resume","--retire-uncommitted",*self.register_args(backup))
        self.assertEqual(before,self.capture())

    def test_retry_modes_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            self.invoke("--execute","--resume","--retry-failed","--retry-failed-and-pending")

    def test_interrupted_partial_preserved_then_regenerated(self):
        self.run_some(1)
        source=self.out/"records/i0000/w0001.part";source.mkdir()
        (source/"phi_cdf.npy").write_bytes(b"incomplete")
        backup=self.change_binary()
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.assertEqual(self.invoke("--execute","--resume","--retire-uncommitted",
                                        "--retry-failed-and-pending",*self.register_args(backup)),0)
        archived=list((self.out/"retired_outputs/i0000_w0001").glob("*/w0001.part/phi_cdf.npy"))
        self.assertEqual(len(archived),1)
        self.assertEqual(archived[0].read_bytes(),b"incomplete")
        self.assertTrue((self.out/"receipts/i0000_w0001.json").exists())

    def test_unreceipted_final_regenerated_not_adopted(self):
        self.run_some(1)
        # Mock a crash between final directory commit and receipt write.
        c=batch.normalized_config(self.raw)
        p=self.out/"records/i0000/w0001"
        self.mock_generate(batch.command(self.exe,p,c,0,1))
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate) as child:
            self.assertEqual(self.invoke("--execute","--resume","--retire-uncommitted"),0)
            self.assertEqual(child.call_count,3)
        self.assertTrue(list((self.out/"retired_outputs/i0000_w0001").glob("*/w0001/metadata.json")))

    def test_interrupted_partial_requires_explicit_option(self):
        self.run_some(1);source=self.out/"records/i0000/w0001.part";source.mkdir()
        (source/"x").write_bytes(b"preserve")
        with self.assertRaises(ValueError):self.invoke("--execute","--resume")
        self.assertEqual((source/"x").read_bytes(),b"preserve")

    def test_retirement_does_not_touch_receipted_data(self):
        self.run_some(1);before=(self.out/"records/i0000/w0000/metadata.json").read_bytes()
        source=self.out/"records/i0000/w0000.part";source.mkdir()
        with self.assertRaises(ValueError):
            self.invoke("--execute","--resume","--retire-uncommitted")
        self.assertEqual((self.out/"records/i0000/w0000/metadata.json").read_bytes(),before)
        self.assertTrue(source.exists())

    def test_retirement_crash_after_move_is_replayed(self):
        self.run_some(1);source=self.out/"records/i0000/w0001.part";source.mkdir()
        (source/"x").write_bytes(b"partial")
        def crash(src,dst):
            src.rename(dst)
            raise KeyboardInterrupt()
        with patch.object(batch,"rename_with_retry",side_effect=crash):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke("--execute","--resume","--retire-uncommitted")
        self.assertFalse(source.exists())
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.assertEqual(self.invoke("--execute","--resume","--retire-uncommitted"),0)
        journals=list((self.out/"retired_outputs/i0000_w0001").glob("*/retirement.json"))
        self.assertEqual(json.loads(journals[0].read_text())["state"],"retired")

    def test_repeated_interruption_preserves_separate_archives(self):
        self.run_some(1)
        signature=json.loads((self.out/"batch_plan.json").read_text())["signature"]
        source=self.out/"records/i0000/w0001.part"
        for payload in (b"first",b"second"):
            source.mkdir();(source/"x").write_bytes(payload)
            gen.retire_uncommitted(self.out,0,1,{signature},batch.rename_with_retry,enabled=True)
        copies=list((self.out/"retired_outputs/i0000_w0001").glob("*/w0001.part/x"))
        self.assertEqual({p.read_bytes() for p in copies},{b"first",b"second"})

    def test_quarantine_never_follows_symlink(self):
        self.run_some(1)
        source=self.out/"records/i0000/w0001.part";source.mkdir()
        outside=self.root/"important";outside.write_bytes(b"keep")
        try:(source/"data").symlink_to(outside)
        except OSError:self.skipTest("symlink unavailable")
        with self.assertRaises(ValueError):
            self.invoke("--execute","--resume","--retire-uncommitted")
        self.assertEqual(outside.read_bytes(),b"keep")

    def test_registry_changed_parent_rejected(self):
        self.run_some(1);backup=self.change_binary()
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):
            self.invoke("--execute","--resume",*self.register_args(backup))
        h=self.history();h["generations"][1]["parent_signature"]="a"*64
        (self.out/"generation_history.json").write_text(json.dumps(h))
        with self.assertRaisesRegex(ValueError,"chain"):
            self.invoke("--execute","--resume")

    def test_failure_unknown_attempt_generation_rejected(self):
        self.make_fail_one()
        p=self.out/"failures/i0000_w0000.json";f=json.loads(p.read_text())
        f["attempts"][0]["generation_signature"]="a"*64;p.write_text(json.dumps(f))
        with self.assertRaisesRegex(ValueError,"Malformed failure"):
            self.invoke("--execute","--resume","--continue-on-error")

    def test_changed_original_base_plan_rejected(self):
        self.run_some(1)
        p=self.out/"batch_plan.json";value=json.loads(p.read_text())
        value["binary_sha256"]["generator"]="0"*64;p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"signature"):
            self.invoke("--execute","--resume")
