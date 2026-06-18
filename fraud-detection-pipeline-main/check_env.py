"""
check_env.py
------------
Pre-flight environment checker for the Fraud Detection Data Pipeline.

Verifies:
  1. Docker Desktop is running (docker info)
  2. WSL2 is enabled and available (Windows only)
  3. Required TCP ports are free: 2181 (Zookeeper), 9092 (Kafka), 8080 (Spark UI)
  4. Required Python packages inside the active .venv: kafka-python, pyspark, pandas
  5. Java 17 / JAVA_HOME is configured

Venv detection logic:
  - Looks for .venv/Scripts/python.exe (Windows) or .venv/bin/python (Linux/macOS)
    relative to the script's own directory, then CWD, then walks up to the repo root.
  - Falls back to the currently-running interpreter if no .venv is found.
  - Package presence is checked by calling `pip show` inside the resolved interpreter
    so results are always scoped to that environment, never system Python.

Usage:
    python check_env.py
    python check_env.py --venv /path/to/custom/venv
"""

import importlib
import importlib.util
import io
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field

# Force UTF-8 output so Unicode symbols render on any Windows terminal
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from typing import List


# ---------------------------------------------------------------------------
# ANSI colour helpers (disabled automatically on Windows without ANSI support)
# ---------------------------------------------------------------------------
def _supports_colour() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


GREEN  = "\033[92m" if _supports_colour() else ""
RED    = "\033[91m" if _supports_colour() else ""
YELLOW = "\033[93m" if _supports_colour() else ""
CYAN   = "\033[96m" if _supports_colour() else ""
BOLD   = "\033[1m"  if _supports_colour() else ""
RESET  = "\033[0m"  if _supports_colour() else ""

TICK  = f"{GREEN}[OK]{RESET}"
CROSS = f"{RED}[FAIL]{RESET}"
WARN  = f"{YELLOW}[WARN]{RESET}"


def ok(msg: str)   -> str: return f"  {TICK}   {msg}"
def fail(msg: str) -> str: return f"  {CROSS} {RED}{msg}{RESET}"
def warn(msg: str) -> str: return f"  {WARN} {YELLOW}{msg}{RESET}"
def info(msg: str) -> str: return f"         {CYAN}{msg}{RESET}"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    notes: List[str] = field(default_factory=list)


results: List[CheckResult] = []


def record(name: str, passed: bool, detail: str = "", notes: List[str] = None) -> CheckResult:
    r = CheckResult(name, passed, detail, notes or [])
    results.append(r)
    return r


# ---------------------------------------------------------------------------
# 1. Docker Desktop
# ---------------------------------------------------------------------------
def check_docker() -> CheckResult:
    print(f"\n{BOLD}[1/5] Docker Desktop{RESET}")

    if shutil.which("docker") is None:
        print(fail("docker CLI not found on PATH"))
        return record("Docker", False, "docker CLI not found")

    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode == 0:
            version = proc.stdout.strip()
            print(ok(f"Docker Desktop is running  (server version: {version})"))
            return record("Docker", True, f"Server version {version}")
        else:
            err = proc.stderr.strip().splitlines()[0] if proc.stderr.strip() else "unknown error"
            print(fail(f"Docker daemon not reachable: {err}"))
            print(info("→ Make sure Docker Desktop is started and the engine is running."))
            return record("Docker", False, err)
    except subprocess.TimeoutExpired:
        print(fail("docker info timed out — Docker Desktop may be starting up"))
        return record("Docker", False, "Timeout")
    except FileNotFoundError:
        print(fail("docker binary not executable"))
        return record("Docker", False, "FileNotFoundError")


# ---------------------------------------------------------------------------
# 2. WSL2 (Windows only)
# ---------------------------------------------------------------------------
def check_wsl2() -> CheckResult:
    print(f"\n{BOLD}[2/5] WSL2{RESET}")

    if platform.system() != "Windows":
        print(warn(f"WSL2 check skipped — running on {platform.system()}"))
        return record("WSL2", True, "Skipped (non-Windows)")

    # wsl --status gives richer info; wsl --list --verbose is more universally available
    try:
        proc = subprocess.run(
            ["wsl", "--list", "--verbose"],
            capture_output=True, timeout=10
        )
        # wsl outputs UTF-16-LE on Windows; decode safely
        try:
            output = proc.stdout.decode("utf-16-le", errors="replace")
        except Exception:
            output = proc.stdout.decode("utf-8", errors="replace")

        lines = [l.strip() for l in output.splitlines() if l.strip()]

        wsl2_distros = [l for l in lines if "2" in l]   # VERSION column contains "2"
        if wsl2_distros:
            print(ok(f"WSL2 is enabled — {len(wsl2_distros)} distro(s) using WSL2"))
            for d in wsl2_distros[:5]:
                print(info(d))
            return record("WSL2", True, f"{len(wsl2_distros)} WSL2 distro(s) found")
        else:
            # Check if any distro exists at all
            if len(lines) > 1:  # first line is header
                print(warn("WSL is installed but no WSL2 distros detected."))
                print(info("→ Run: wsl --set-default-version 2"))
                return record("WSL2", False, "No WSL2 distros")
            else:
                print(fail("WSL has no distros installed."))
                print(info("→ Run: wsl --install"))
                return record("WSL2", False, "WSL not configured")

    except FileNotFoundError:
        print(fail("wsl.exe not found — WSL is not installed"))
        print(info("→ Run in PowerShell (admin): wsl --install"))
        return record("WSL2", False, "wsl.exe not found")
    except subprocess.TimeoutExpired:
        print(warn("wsl --list timed out"))
        return record("WSL2", False, "Timeout")


# ---------------------------------------------------------------------------
# 3. Port availability
# ---------------------------------------------------------------------------
REQUIRED_PORTS = {
    2181: "Zookeeper",
    9092: "Kafka",
    8080: "Spark UI",
}


def _port_is_free(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """Return True if the port is NOT currently bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        result = s.connect_ex((host, port))
        return result != 0   # non-zero → connection refused → port is free


def check_ports() -> CheckResult:
    print(f"\n{BOLD}[3/5] Port Availability{RESET}")
    all_free = True

    for port, service in REQUIRED_PORTS.items():
        if _port_is_free(port):
            print(ok(f"Port {port:>5}  ({service}) is FREE"))
        else:
            print(fail(f"Port {port:>5}  ({service}) is IN USE"))
            print(info(f"→ Find the process: netstat -ano | findstr :{port}"))
            all_free = False

    status = "All ports free" if all_free else "One or more ports in use"
    return record("Ports", all_free, status)


# ---------------------------------------------------------------------------
# 4. Python packages (venv-aware)
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    "kafka-python",
    "pyspark",
    "pandas",
    "faker",
]


def _find_venv_python(start_dir: str) -> tuple[str, str]:
    """
    Walk upward from *start_dir* looking for a .venv directory.
    Returns (python_executable_path, venv_root) or (sys.executable, "") if not found.

    Search order per directory:
      Windows : .venv/Scripts/python.exe
      POSIX   : .venv/bin/python3, .venv/bin/python
    """
    candidates_rel = (
        [os.path.join(".venv", "Scripts", "python.exe")]  # Windows
        if platform.system() == "Windows"
        else [
            os.path.join(".venv", "bin", "python3"),
            os.path.join(".venv", "bin", "python"),
        ]
    )

    current = os.path.abspath(start_dir)
    for _ in range(6):  # climb at most 6 levels (stops at drive root)
        for rel in candidates_rel:
            candidate = os.path.join(current, rel)
            if os.path.isfile(candidate):
                venv_root = os.path.join(current, ".venv")
                return candidate, venv_root
        parent = os.path.dirname(current)
        if parent == current:  # reached filesystem root
            break
        current = parent

    return sys.executable, ""  # fall back to the running interpreter


def _pip_show(python_exe: str, package: str) -> tuple[bool, str]:
    """
    Call `<python_exe> -m pip show <package>` and return (found, version).
    Using pip show is more reliable than importlib for packages whose import
    name differs from the pip name (e.g. kafka-python → import kafka).
    """
    try:
        proc = subprocess.run(
            [python_exe, "-m", "pip", "show", package],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode == 0:
            version = "n/a"
            for line in proc.stdout.splitlines():
                if line.lower().startswith("version:"):
                    version = line.split(":", 1)[1].strip()
                    break
            return True, version
        return False, ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, ""


def check_packages(venv_python: str = "") -> CheckResult:
    """
    Check that every required package is installed inside the resolved venv.

    Args:
        venv_python: Path to the venv Python executable (resolved by main).
                     If empty, _find_venv_python() is called here.
    """
    # ---- Resolve which Python to interrogate --------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not venv_python:
        venv_python, venv_root = _find_venv_python(script_dir)
    else:
        venv_root = os.path.dirname(os.path.dirname(venv_python))  # .venv/

    in_venv    = bool(venv_root)
    is_current = os.path.abspath(venv_python) == os.path.abspath(sys.executable)

    # ---- Header -------------------------------------------------------------
    py_ver_proc = subprocess.run(
        [venv_python, "--version"], capture_output=True, text=True, timeout=10
    )
    py_ver = py_ver_proc.stdout.strip() or py_ver_proc.stderr.strip()

    print(f"\n{BOLD}[4/5] Python Packages{RESET}")

    if in_venv:
        print(ok(f"Virtual environment detected"))
        print(info(f"Venv root  : {venv_root}"))
        print(info(f"Interpreter: {venv_python}"))
        print(info(f"Version    : {py_ver}"))
    else:
        if is_current:
            print(warn("No .venv found — checking current interpreter (system Python)"))
        else:
            print(warn(f"No .venv found — falling back to: {venv_python}"))
        print(info("Create one with: python -m venv .venv  then  pip install -r requirements.txt"))

    # ---- Check each package via pip show ------------------------------------
    print()
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        found, version = _pip_show(venv_python, pkg)
        if found:
            env_tag = "(venv)" if in_venv else "(system)"
            print(ok(f"{pkg:<15}  v{version}  {CYAN}{env_tag}{RESET}"))
        else:
            print(fail(f"{pkg:<15}  NOT installed"))
            activate = (
                ".venv\\Scripts\\activate" if platform.system() == "Windows"
                else "source .venv/bin/activate"
            )
            print(info(f"Activate venv first: {activate}"))
            print(info(f"Then run: pip install {pkg}"))
            all_ok = False

    status = "All packages present" if all_ok else "Missing packages"
    if not in_venv:
        status += " (system Python, no .venv)"
    return record("Packages", all_ok, status)


# ---------------------------------------------------------------------------
# 5. Java 17 / JAVA_HOME
# ---------------------------------------------------------------------------
def check_java() -> CheckResult:
    print(f"\n{BOLD}[5/5] Java 17 / JAVA_HOME{RESET}")
    issues = []

    # --- JAVA_HOME env var ---
    java_home = os.environ.get("JAVA_HOME", "")
    if java_home:
        print(ok(f"JAVA_HOME is set  ->  {java_home}"))
    else:
        print(fail("JAVA_HOME is NOT set"))
        print(info("Run set_java_env.ps1 as Administrator to fix this."))
        issues.append("JAVA_HOME missing")

    # --- JAVA_HOME points to a real JDK 17 dir ---
    if java_home:
        java_exe = os.path.join(java_home, "bin", "java.exe")
        if os.path.isfile(java_exe):
            print(ok(f"java.exe found at  {java_exe}"))
        else:
            print(fail(f"java.exe NOT found under JAVA_HOME  ({java_exe})"))
            issues.append("java.exe missing under JAVA_HOME")

    # --- java -version ---
    try:
        proc = subprocess.run(
            ["java", "-version"],
            capture_output=True, text=True, timeout=10
        )
        # java -version prints to stderr by convention
        version_output = (proc.stderr or proc.stdout).strip().splitlines()
        version_line = version_output[0] if version_output else ""

        if version_line:
            print(ok(f"java -version  ->  {version_line}"))
        else:
            print(fail("java -version produced no output"))
            issues.append("java -version empty")

        if "17." in version_line or '"17' in version_line:
            print(ok("Java 17 confirmed!"))
        elif version_line:
            print(warn(f"Java found but may not be v17 — check output above"))
            issues.append("Version may not be 17")
    except FileNotFoundError:
        print(fail("java not found on PATH"))
        print(info("Run set_java_env.ps1 (as Administrator) then open a new terminal."))
        issues.append("java not on PATH")
    except subprocess.TimeoutExpired:
        print(fail("java -version timed out"))
        issues.append("Timeout")

    passed  = len(issues) == 0
    detail  = "JDK 17 ready" if passed else "; ".join(issues)
    return record("Java 17", passed, detail)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary():
    print(f"\n{'=' * 52}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{'=' * 52}")
    all_passed = True
    for r in results:
        icon = TICK if r.passed else CROSS
        label = f"{r.name:<12}"
        detail = r.detail[:35] if r.detail else ""
        print(f"  {icon}  {label}  {detail}")
        if not r.passed:
            all_passed = False
    print(f"{'=' * 52}")
    if all_passed:
        print(f"\n{GREEN}{BOLD}  [OK] All checks passed -- environment is ready!{RESET}\n")
    else:
        failed = [r.name for r in results if not r.passed]
        print(f"\n{RED}{BOLD}  [FAIL] Fix the items above before running docker compose up:{RESET}")
        for name in failed:
            print(f"      * {name}")
        print()
    return all_passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-flight environment checker for the Fraud Detection pipeline."
    )
    parser.add_argument(
        "--venv",
        metavar="PATH",
        default="",
        help=(
            "Explicit path to a venv Python interpreter "
            "(e.g. .venv/Scripts/python.exe). "
            "If omitted, the script auto-detects .venv/ by walking up from the script directory."
        ),
    )
    args = parser.parse_args()

    # Resolve venv interpreter once so we can show it in the header
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.venv:
        venv_python = args.venv
        venv_root   = os.path.dirname(os.path.dirname(os.path.abspath(args.venv)))
    else:
        venv_python, venv_root = _find_venv_python(script_dir)

    print(f"\n{'*' * 52}")
    print(f"{BOLD}  Fraud Detection Pipeline -- Environment Check{RESET}")
    print(f"  OS: {platform.system()} {platform.release()}  |  Python {sys.version.split()[0]}")
    if venv_root:
        print(f"  Venv: {venv_root}")
    else:
        print(f"  Venv: none detected (using system Python)")
    print(f"{'*' * 52}")

    check_docker()
    check_wsl2()
    check_ports()
    check_packages(venv_python=venv_python)
    check_java()

    passed = print_summary()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
