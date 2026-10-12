"""Orchestration tests. The subprocess is explicitly mocked; not GPU validation."""
from __future__ import annotations
import importlib.util
import contextlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("condition_batch",ROOT/"tools/generate_phase_dataset.py")
batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)


def write_npy(path, shape, values):
    body=repr({"descr":"<f8","fortran_order":False,"shape":shape})
    padding=(- (10+len(body)+1))%64
    header=(body+" "*padding+"\n").encode("latin1")
    path.write_bytes(b"\x93NUMPY\x01\x00"+struct.pack("<H",len(header))+header+struct.pack("<"+"d"*len(values),*values))


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.out=self.root/"data";self.exe=self.root/"rainbow_generate.exe"
        self.exe.write_bytes(b"explicit-test-placeholder")
        (self.root/"modules").mkdir()
        for name in batch.MODULES:(self.root/"modules"/name).write_bytes(name.encode())
        raw=json.loads((ROOT/"configs/dataset_water_a1.json").read_text())
        raw.update(output_root=str(self.out),generator=str(self.exe),reserve_gib=0)
        raw["wavelengths"]={"start_nm":700,"stop_nm":705,"step_nm":5}
        raw["inclinations"]["count"]=2
        raw["solver"].update(grid=9,query_theta=4,query_phi=8,cdf_theta=2,cdf_phi=4)
        self.raw=raw;self.cfg=self.root/"config.json";self.save()
        self.calls=[]
    def save(self):self.cfg.write_text(json.dumps(self.raw))
    def invoke(self,*args):
        with contextlib.redirect_stdout(io.StringIO()):return batch.run(["--config",str(self.cfg),*args])
    def mock_generate(self,cmd,**kwargs):
        self.calls.append(cmd)
        opt={};i=1
        while i<len(cmd):
            if cmd[i] in ("--sphere","--allow-underresolved"):opt[cmd[i]]=True;i+=1
            else:opt[cmd[i]]=cmd[i+1];i+=2
        path=Path(opt["--out"]);path.mkdir(parents=True)
        a=float(opt["--inclination-deg"]);r=math.radians(a)
        k=[math.cos(r),-math.sin(r),0];e=[0,0,1];f=[-math.sin(r),-math.cos(r),0]
        nt=int(opt["--cdf-theta"]);np_=int(opt["--cdf-phi"])
        m={"complete":True,"schema":"rainbow.phase_cdf.numpy.v2","generator_version":batch.GENERATOR_VERSION,
           "coordinate_contract_id":"rainbow.phase_cdf.coordinates.v1","direction_convention":"physical_propagation",
           "dtype":"<f8","order":"C","density_measure":"solid_angle_sr","cdf_axis_order":["phi_cell","theta_edge"],
           "theta_count":nt,"phi_count":np_,"query_grid":[int(opt["--query-theta"]),int(opt["--query-phi"])],
           "incident_grid":[int(opt["--grid"])]*2,"stage":opt["--stage"],"force_sphere":opt.get("--sphere",False),
           "focal_quarter_turn_offsets":list(map(int,opt["--focal-offsets"].split(','))),
           "radius_mm":batch.fp32(float(opt["--radius-mm"])),"wavelength_nm":batch.fp32(float(opt["--wavelength-nm"])),
           "incident_inclination_degrees":a,"material":{"model":"water_iapws_r9_97","temperature_kelvin":float(opt["--temperature-c"])+273.15,"pressure_pascal":float(opt["--pressure-pa"])},
           "sampling_frame_columns":[[e[j],f[j],k[j]] for j in range(3)],
           "geometry":{"projected_area_mm2":3.0,"projected_area_converged":True,"convexity_verified":True},
           "cross_sections":{"model":"geometric_nonabsorbing","wave_scattering_cross_section_computed":False,"scattering_mm2":3.0,"extinction_mm2":3.0,"absorption_mm2":0},
           "quality":{"invalid_values":0,"incomplete_directions":0,"allow_underresolved":opt.get("--allow-underresolved",False),"underresolved_directions":0,"cdf_l1_mass_error":0,"cdf_lost_probability_mass":0},
           "storage_processing":{"additional_filter":"none","coarsening_tv":0},"hg":{"target":"saved_cdf","g":0.0}}
        (path/"metadata.json").write_text(json.dumps(m))
        u=[math.sin(math.pi*j/(2*nt))**2 for j in range(nt+1)];u[0]=0;u[-1]=1
        write_npy(path/"u_edges.npy",(nt+1,),u)
        write_npy(path/"phi_cdf.npy",(np_+1,),[j/np_ for j in range(np_+1)])
        write_npy(path/"theta_given_phi_cdf.npy",(np_,nt+1),u*np_)
        return types.SimpleNamespace(returncode=0)
    def run_some(self,limit=None):
        args=["--execute"]+(["--limit",str(limit)] if limit else [])
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):return self.invoke(*args)
    def test_default_dry_run(self):
        self.assertEqual(self.invoke(),0);self.assertFalse(self.out.exists());self.assertEqual(self.calls,[])
    def test_wave_endpoints(self):
        w=batch.wavelengths(dict(start_nm=380,stop_nm=830,step_nm=5));self.assertEqual(len(w),91);self.assertEqual((w[0],w[-1]),("380","830"))
    def test_decimal_steps(self):
        self.assertEqual(batch.wavelengths(dict(start_nm="700.1",stop_nm="700.3",step_nm="0.1")),["700.1","700.2","700.3"])
    def test_wave_bad_step(self):
        for step in (0,-1,3):
            with self.assertRaises(ValueError):batch.wavelengths(dict(start_nm=700,stop_nm=705,step_nm=step))
    def test_fp32_alias(self):
        with self.assertRaises(ValueError):batch.wavelengths(dict(start_nm="700",stop_nm="700.000001",step_nm="0.000001"))
    def test_angle_centers(self):
        values=batch.inclinations(dict(min_degrees=-90,max_degrees=90,count=900,placement="linear_angle_cell_centers"))
        self.assertEqual(len(values),900);self.assertEqual((values[0],values[-1]),("-89.9","89.9"))
    def test_wrong_angle_domain(self):
        with self.assertRaises(ValueError):batch.inclinations(dict(min_degrees=0,max_degrees=180,count=900,placement="linear_angle_cell_centers"))
    def test_grid_refuses_nonnested(self):
        self.raw["solver"]["cdf_phi"]=3
        with self.assertRaises(ValueError):batch.normalized_config(self.raw)
    def test_unknown_config(self):
        self.raw["blur"]=1
        with self.assertRaises(ValueError):batch.normalized_config(self.raw)
    def test_all_jobs_and_resume(self):
        self.assertEqual(self.run_some(),0);self.assertEqual(len(self.calls),4)
        self.assertTrue(json.loads((self.out/"progress.json").read_text())["complete"])
        with patch.object(batch.subprocess,"run",side_effect=AssertionError("must skip")):
            self.assertEqual(self.invoke("--execute","--resume"),0)
    def test_limit_resume(self):
        self.run_some(1);self.assertFalse(json.loads((self.out/"progress.json").read_text())["complete"])
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):self.invoke("--execute","--resume")
        self.assertEqual(len(self.calls),4)
    def test_no_automatic_resume(self):
        self.run_some(1)
        with self.assertRaises(FileExistsError):self.invoke("--execute")
    def test_binary_change_rejected(self):
        self.run_some(1);self.exe.write_bytes(b"different executable")
        with self.assertRaises(ValueError):self.invoke("--execute","--resume")
    def test_configuration_change_rejected(self):
        self.run_some(1);self.raw["pressure_pa"]=100000;self.save()
        with self.assertRaises(ValueError):self.invoke("--execute","--resume")
    def test_truncated_npy_rejected(self):
        self.run_some(1);p=self.out/"records/i0000/w0000/u_edges.npy";p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(ValueError):self.invoke("--execute","--resume")
    def test_metadata_changed_rejected(self):
        self.run_some(1);p=self.out/"records/i0000/w0000/metadata.json";m=json.loads(p.read_text());m["geometry"]["projected_area_mm2"]=-1;p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):self.invoke("--execute","--resume")
    def test_frame_transpose_rejected(self):
        self.run_some(1);p=self.out/"records/i0000/w0000/metadata.json";m=json.loads(p.read_text());m["sampling_frame_columns"]=[list(r) for r in zip(*m["sampling_frame_columns"])];p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):self.invoke("--execute","--resume")
    def test_failed_subprocess_stops(self):
        def bad(cmd,**kw):return types.SimpleNamespace(returncode=1)
        with patch.object(batch.subprocess,"run",side_effect=bad) as mock:
            with self.assertRaises(RuntimeError):self.invoke("--execute")
        self.assertEqual(mock.call_count,1);self.assertFalse((self.out/".batch.lock").exists());self.assertFalse((self.out/"receipts").exists())
    def test_incomplete_directory_preserved(self):
        self.run_some(1);p=self.out/"records/i0000/w0001.part";p.mkdir();(p/"keep").write_text("do not delete")
        with self.assertRaises(ValueError):self.invoke("--execute","--resume")
        self.assertTrue((p/"keep").exists())
    def test_lock_preserved(self):
        self.out.mkdir();lock=self.out/".batch.lock";lock.write_text("another writer")
        with self.assertRaises(RuntimeError):self.invoke("--execute")
        self.assertEqual(lock.read_text(),"another writer")
    def test_explicit_warning_flag_passed(self):
        with patch.object(batch.subprocess,"run",side_effect=self.mock_generate):self.invoke("--execute","--allow-underresolved","--limit","1")
        self.assertIn("--allow-underresolved",self.calls[0])
    def test_missing_receipt_not_adopted(self):
        self.run_some(1);(self.out/"receipts/i0000_w0000.json").unlink()
        with self.assertRaises(ValueError):self.invoke("--execute","--resume")
    def test_missing_module(self):
        (self.root/"modules"/batch.MODULES[0]).unlink()
        with self.assertRaises(FileNotFoundError):self.invoke("--execute")
    def test_no_disk_space(self):
        with patch.object(batch.shutil,"disk_usage",return_value=types.SimpleNamespace(free=0)):
            with self.assertRaises(RuntimeError):self.invoke("--execute")
        self.assertFalse((self.out/"records").exists())
    def test_wrong_npy_axis(self):
        self.run_some(1);p=self.out/"records/i0000/w0000/theta_given_phi_cdf.npy";write_npy(p,(3,4),[0]*12)
        with self.assertRaises(ValueError):self.invoke("--execute","--resume")
    def test_nonfinite_json_rejected(self):
        self.cfg.write_text('{"a": NaN}')
        with self.assertRaises(ValueError):batch.load_json(self.cfg)




# A log-format fixture, NOT a numerical simulation or ground-truth computation.
CDF_REJECTION = """rainbow_generate: CDF input rejected: max_intensity=6.62286, invalid_values=653, incomplete=653, underresolved=43200, malformed=0, first_problem=1644082, L1=0, lost_mass=0, weight_underflow=0
[cdf-input] stage=diffraction directions=6480000 allow_underresolved=1 CDF_accumulation_started=0
[cdf-input] stage=incoherent invalid_values=0 incomplete=1 underresolved=0 first_hard_problem=1687286
[cdf-input] optical flags=64 query_flags=48 known_hits_complete=0 hit_count=6 evaluated_hits=2 rejected_hits=4 refinement_hits=0 boundary_hits=4 singular_hits=0 folded_evaluated_hits=0 first_problem_patch_id=32956688
[cdf-input] focal flags=2 valid=0 corrected_hits=2 first_problem_patch_id=32956688 intensity_s=nan intensity_p=nan family_hits=1,1,0,0
"""


class FailureTests(unittest.TestCase):
    # Reuse fixtures only, without inheriting/rerunning the original test cases.
    setUp = BatchTests.setUp
    save = BatchTests.save
    invoke = BatchTests.invoke
    mock_generate = BatchTests.mock_generate
    run_some = BatchTests.run_some

    def fail_job(self, cmd, **kwargs):
        self.calls.append(cmd)
        kwargs["stdout"].write(CDF_REJECTION.encode())
        return types.SimpleNamespace(returncode=1)

    def failures_then_success(self, failed_calls):
        counter = 0
        def action(cmd, **kwargs):
            nonlocal counter
            current = counter
            counter += 1
            return (self.fail_job if current in failed_calls else self.mock_generate)(cmd, **kwargs)
        return action

    def state(self):
        return json.loads((self.out/"progress.json").read_text())

    def log_fixture(self, text=CDF_REJECTION):
        log = self.root/"fixture.log"
        log.write_text(text)
        return log

    def test_recognizes_supplied_failure_format(self):
        detail = batch.inspect_job_failure(self.log_fixture(), 1)
        self.assertTrue(detail["continuable"])
        self.assertEqual(detail["category"], "cdf_input_boundary_pending")
        self.assertIn("intensity_s=nan", detail["cdf_diagnostic_lines"][-1])
        json.dumps(detail, allow_nan=False)

    def test_rejects_signal_or_success_codes(self):
        for rc in (0, -9, 139, 3221225477):
            self.assertFalse(batch.inspect_job_failure(self.log_fixture(), rc)["continuable"])

    def test_underresolution_only_is_not_deferred(self):
        text = CDF_REJECTION.replace("invalid_values=653, incomplete=653", "invalid_values=0, incomplete=0")
        self.assertFalse(batch.inspect_job_failure(self.log_fixture(text), 1)["continuable"])

    def test_requires_pre_accumulation_marker(self):
        for text in (CDF_REJECTION.replace("CDF_accumulation_started=0", "CDF_accumulation_started=1"),
                     "\n".join(line for line in CDF_REJECTION.splitlines() if "directions=6480000" not in line)):
            self.assertFalse(batch.inspect_job_failure(self.log_fixture(text), 1)["continuable"])

    def test_missing_rejection_header_is_fatal(self):
        text = "\n".join(CDF_REJECTION.splitlines()[1:])
        self.assertFalse(batch.inspect_job_failure(self.log_fixture(text), 1)["continuable"])

    def test_log_tail_has_memory_limit_and_fails_closed(self):
        log = self.log_fixture(CDF_REJECTION + "x"*(300 << 10))
        self.assertFalse(batch.inspect_job_failure(log, 1)["continuable"])

    def test_real_subprocess_log_transport(self):
        # An actual child process emits the fixture and exits 1. No GPU involved.
        import subprocess, sys
        log = self.root/"child.log"
        with log.open("wb") as stream:
            result = subprocess.run([sys.executable, "-c", "import sys;sys.stdout.write("+repr(CDF_REJECTION)+");sys.exit(1)"],
                                    stdout=stream, stderr=subprocess.STDOUT, check=False)
        self.assertTrue(batch.inspect_job_failure(log, result.returncode)["continuable"])

    def test_defer_then_continue_without_accepting_bad_data(self):
        with patch.object(batch.subprocess, "run", side_effect=self.failures_then_success({0})):
            self.assertEqual(self.invoke("--execute", "--continue-on-error"), 2)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.state()["completed"], 3)
        self.assertEqual(self.state()["failed"], 1)
        self.assertEqual(self.state()["unattempted"], 0)
        self.assertFalse(self.state()["complete"])
        self.assertFalse((self.out/"records/i0000/w0000").exists())
        self.assertFalse((self.out/"receipts/i0000_w0000.json").exists())
        failures = json.loads((self.out/"failures.json").read_text())
        self.assertEqual(failures["count"], 1)
        self.assertEqual(failures["jobs"][0]["job"], "i0000_w0000")

    def test_original_stop_policy_remains_default(self):
        with patch.object(batch.subprocess, "run", side_effect=self.fail_job):
            with self.assertRaises(RuntimeError): self.invoke("--execute")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.state()["completed"], 0)
        self.assertTrue((self.out/"failures/i0000_w0000.json").exists())

    def test_resume_skips_only_known_failure_when_explicit(self):
        with patch.object(batch.subprocess, "run", side_effect=self.failures_then_success({0})):
            self.assertEqual(self.invoke("--execute", "--continue-on-error", "--limit", "2"), 2)
        with patch.object(batch.subprocess, "run", side_effect=self.mock_generate):
            self.assertEqual(self.invoke("--execute", "--continue-on-error", "--resume"), 2)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.state()["completed"], 3)
        self.assertEqual(self.state()["failed"], 1)

    def test_resume_without_continue_does_not_silently_skip(self):
        with patch.object(batch.subprocess, "run", side_effect=self.fail_job):
            self.invoke("--execute", "--continue-on-error", "--limit", "1")
        with self.assertRaises(RuntimeError): self.invoke("--execute", "--resume")

    def test_retries_failures_only_and_preserves_history(self):
        with patch.object(batch.subprocess, "run", side_effect=self.failures_then_success({0})):
            self.invoke("--execute", "--continue-on-error")
        original = (self.out/"logs/i0000_w0000.log").read_bytes()
        with patch.object(batch.subprocess, "run", side_effect=self.mock_generate) as run:
            self.assertEqual(self.invoke("--execute", "--resume", "--retry-failed"), 0)
            self.assertEqual(run.call_count, 1)
        self.assertTrue(self.state()["complete"])
        self.assertEqual(self.state()["failed"], 0)
        history=json.loads((self.out/"failures/i0000_w0000.json").read_text())
        self.assertEqual(history["state"], "resolved")
        self.assertEqual(len(history["attempts"]), 1)
        self.assertEqual((self.out/"logs/i0000_w0000.log").read_bytes(), original)
        self.assertEqual(len(list((self.out/"logs").glob("i0000_w0000*.log"))), 2)

    def test_retry_does_not_run_unattempted_jobs(self):
        with patch.object(batch.subprocess, "run", side_effect=self.fail_job):
            self.invoke("--execute", "--continue-on-error", "--limit", "1")
        with patch.object(batch.subprocess, "run", side_effect=self.mock_generate) as run:
            self.assertEqual(self.invoke("--execute", "--resume", "--retry-failed"), 0)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(self.state()["completed"], 1)
        self.assertEqual(self.state()["unattempted"], 3)
        self.assertFalse(self.state()["complete"])

    def test_failed_retry_is_not_an_infinite_loop(self):
        with patch.object(batch.subprocess, "run", side_effect=self.fail_job):
            self.invoke("--execute", "--continue-on-error", "--limit", "1")
        with patch.object(batch.subprocess, "run", side_effect=self.fail_job) as run:
            self.assertEqual(self.invoke("--execute", "--resume", "--retry-failed", "--continue-on-error"), 2)
            self.assertEqual(run.call_count, 1)
        history=json.loads((self.out/"failures/i0000_w0000.json").read_text())
        self.assertEqual(len(history["attempts"]), 2)

    def test_total_failure_limit_stops(self):
        with patch.object(batch.subprocess, "run", side_effect=self.fail_job):
            with self.assertRaises(RuntimeError):
                self.invoke("--execute", "--continue-on-error", "--max-job-failures", "1")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.state()["status"], "stopped_error")

    def test_consecutive_failure_limit_stops(self):
        with patch.object(batch.subprocess, "run", side_effect=self.fail_job):
            with self.assertRaises(RuntimeError):
                self.invoke("--execute", "--continue-on-error", "--max-consecutive-failures", "2")
        self.assertEqual(len(self.calls), 2)

    def test_success_resets_consecutive_counter(self):
        with patch.object(batch.subprocess, "run", side_effect=self.failures_then_success({0, 2})):
            self.assertEqual(self.invoke("--execute", "--continue-on-error", "--max-consecutive-failures", "2"), 2)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.state()["failed"], 2)

    def test_unrecognized_exit_always_stops(self):
        def crash(cmd, **kwargs):
            self.calls.append(cmd)
            kwargs["stdout"].write(b"CUDA out of memory\n")
            return types.SimpleNamespace(returncode=1)
        with patch.object(batch.subprocess, "run", side_effect=crash):
            with self.assertRaises(RuntimeError): self.invoke("--execute", "--continue-on-error")
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(RuntimeError): self.invoke("--execute", "--resume", "--continue-on-error")

    def test_failure_with_part_is_preserved_and_stops(self):
        def partial(cmd, **kwargs):
            out=Path(cmd[cmd.index("--out")+1]+".part")
            out.mkdir();(out/"keep").write_text("never delete")
            return self.fail_job(cmd, **kwargs)
        with patch.object(batch.subprocess, "run", side_effect=partial):
            with self.assertRaises(RuntimeError): self.invoke("--execute", "--continue-on-error")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual((self.out/"records/i0000/w0000.part/keep").read_text(), "never delete")
        with self.assertRaises(ValueError): self.invoke("--execute", "--resume", "--continue-on-error")

    def test_failure_with_final_output_is_not_adopted(self):
        def partial(cmd, **kwargs):
            self.mock_generate(cmd, **kwargs)
            kwargs["stdout"].write(CDF_REJECTION.encode())
            return types.SimpleNamespace(returncode=1)
        with patch.object(batch.subprocess, "run", side_effect=partial):
            with self.assertRaises(RuntimeError): self.invoke("--execute", "--continue-on-error")
        self.assertTrue((self.out/"records/i0000/w0000").exists())
        self.assertFalse((self.out/"receipts/i0000_w0000.json").exists())

    def test_successful_exit_with_bad_output_still_stops(self):
        def malformed(cmd, **kwargs):
            r=self.mock_generate(cmd, **kwargs)
            out=Path(cmd[cmd.index("--out")+1]);(out/"u_edges.npy").write_bytes(b"bad")
            return r
        with patch.object(batch.subprocess, "run", side_effect=malformed):
            with self.assertRaises(ValueError): self.invoke("--execute", "--continue-on-error")
        self.assertEqual(len(self.calls), 1)

    def test_binary_change_is_not_a_retry_bypass(self):
        with patch.object(batch.subprocess, "run", side_effect=self.fail_job):
            self.invoke("--execute", "--continue-on-error", "--limit", "1")
        self.exe.write_bytes(b"modified")
        with self.assertRaises(ValueError): self.invoke("--execute", "--resume", "--retry-failed", "--continue-on-error")

    def test_failure_ledger_signature_is_checked(self):
        with patch.object(batch.subprocess, "run", side_effect=self.fail_job):
            self.invoke("--execute", "--continue-on-error", "--limit", "1")
        p=self.out/"failures/i0000_w0000.json";v=json.loads(p.read_text());v["signature"]="bad";p.write_text(json.dumps(v))
        with self.assertRaises(ValueError): self.invoke("--execute", "--resume", "--continue-on-error")

    def test_retry_requires_resume(self):
        with self.assertRaises(ValueError): self.invoke("--retry-failed")

    def test_failure_limits_must_be_positive(self):
        for option in ("--max-job-failures", "--max-consecutive-failures"):
            with self.assertRaises(ValueError): self.invoke(option, "0")

    def test_continue_dry_run_writes_nothing(self):
        self.assertEqual(self.invoke("--continue-on-error", "--dry-run"), 0)
        self.assertFalse(self.out.exists())

    def test_receipt_commit_before_failure_resolution_is_reconciled(self):
        with patch.object(batch.subprocess, "run", side_effect=self.failures_then_success({0})):
            self.invoke("--execute", "--continue-on-error")
        with patch.object(batch.subprocess, "run", side_effect=self.mock_generate):
            self.invoke("--execute", "--resume", "--retry-failed")
        path=self.out/"failures/i0000_w0000.json";v=json.loads(path.read_text());v["state"]="unresolved";path.write_text(json.dumps(v))
        with patch.object(batch.subprocess, "run", side_effect=AssertionError("must not rerun")):
            self.assertEqual(self.invoke("--execute", "--resume"), 0)
        self.assertEqual(json.loads(path.read_text())["state"], "resolved")

if __name__=="__main__":unittest.main()
