#!/usr/bin/env bash

# Install / bootstrap a tvss_nav workspace under ~/lisn_ws
# - Only the tvss_nav repo needs to be cloned manually.
# - This script will clone tvss_nav + dynamic_obstacle_detector + sfm_local_controller + lightsfm
#   into a fresh catkin workspace and build everything.
#
# Configuration (can be overridden via env vars before running):
#   LISN_WS_DIR           : workspace root, default "$HOME/lisn_ws"
#   LISN_REMOTE           : tvss_nav git URL
#   LISN_DOD_REMOTE       : dynamic_obstacle_detector fork URL
#   LISN_SFM_REMOTE       : sfm_local_controller fork URL
#   LISN_MSGS_REMOTE      : tvsn_msgs git URL
#   LISN_LIGHTSFM_REMOTE  : lightsfm fork URL
#   LISN_ARENA_EVAL_BRANCH : arena_evaluation branch name, default "master"

set -euo pipefail

log()  { printf '[lisn] %s\n' "$*" >&2; }
err()  { printf '[lisn][ERROR] %s\n' "$*" >&2; }
warn() { printf '[lisn][WARN] %s\n' "$*" >&2; }

# Execution toggles (override via environment when needed)
LISN_SKIP_FETCH=${LISN_SKIP_FETCH:-0}
LISN_SKIP_ROSDEP=${LISN_SKIP_ROSDEP:-0}
LISN_SKIP_LIGHTSFM_BUILD=${LISN_SKIP_LIGHTSFM_BUILD:-0}
LISN_SKIP_BUILD=${LISN_SKIP_BUILD:-0}
LISN_FORCE=${LISN_FORCE:-1}
LISN_SKIP_ARENA=${LISN_SKIP_ARENA:-0}

CPU_COUNT="$(if command -v nproc >/dev/null 2>&1; then nproc; else echo 4; fi)"
LISN_PARALLEL_JOBS=${LISN_PARALLEL_JOBS:-$CPU_COUNT}

# Workspace root
WS_DIR="${LISN_WS_DIR:-$HOME/lisn_ws}"

# ROS distro (default:noetic)
ROS_DISTRO="${ROS_DISTRO:-noetic}"

# Default remotes (override via env if needed)
LISN_REMOTE="${LISN_REMOTE:-https://github.com/Social-Nav/tvss_nav.git}"
LISN_DOD_REMOTE="${LISN_DOD_REMOTE:-https://github.com/Social-Nav/dynamic_obstacle_detector.git}"
LISN_SFM_REMOTE="${LISN_SFM_REMOTE:-https://github.com/Social-Nav/sfm_local_controller.git}"
LISN_MSGS_REMOTE="${LISN_MSGS_REMOTE:-https://github.com/Social-Nav/tvsn_msgs.git}"
LISN_LIGHTSFM_REMOTE="${LISN_LIGHTSFM_REMOTE:-https://github.com/Social-Nav/lightsfm.git}"
LISN_ARENA_REMOTE="${LISN_ARENA_REMOTE:-https://github.com/Arena-Rosnav/arena-rosnav.git}"
LISN_ARENA_COMMIT="${LISN_ARENA_COMMIT:-6ad00193b17cccf160753b97da950b49ca0371c7}"
LISN_SIM_SETUP_REMOTE="${LISN_SIM_SETUP_REMOTE:-https://github.com/Social-Nav/arena-simulation-setup.git}"
LISN_ARENA_EVAL_REMOTE="${LISN_ARENA_EVAL_REMOTE:-https://github.com/Social-Nav/arena-evaluation.git}"
LISN_ARENA_EVAL_BRANCH="${LISN_ARENA_EVAL_BRANCH:-master}"

if [[ -z "${LISN_REMOTE}" ]]; then
  err "LISN_REMOTE is not set. Please set a git URL for the tvss_nav package."
  exit 1
fi

# Prepare workspace directory
log "Using workspace directory: ${WS_DIR}"
if [[ -d "${WS_DIR}" && -n "$(ls -A "${WS_DIR}" 2>/dev/null)" && ${LISN_FORCE} -ne 1 ]]; then
  err "Workspace directory ${WS_DIR} is not empty. Set LISN_FORCE=1 to reuse it."
  exit 1
fi
mkdir -p "${WS_DIR}/src"
cd "${WS_DIR}"

# Mark existing repos as safe for git (avoid 'dubious ownership' when workspace is mounted)
for d in src/*; do
  if [[ -d "${d}/.git" ]]; then
    git config --global --add safe.directory "${WS_DIR}/${d}" || true
  fi
done

# Helper: clone or update git repo
clone_or_update() {
  local target="$1"
  local url="$2"

  if [[ ${LISN_SKIP_FETCH} -eq 1 ]]; then
    if [[ -d "${target}/.git" ]]; then
      log "Reusing existing repo: ${target} (LISN_SKIP_FETCH=1)"
      return 0
    fi
    warn "Skipping fetch for ${target} because LISN_SKIP_FETCH=1 and no repo exists"
    return 0
  fi

  mkdir -p "$(dirname "${target}")"

  if [[ -d "${target}/.git" ]]; then
    log "Updating existing repo: ${target}"
    git -C "${target}" fetch --all --prune >/dev/null 2>&1 || warn "git fetch failed for ${target}"
    git -C "${target}" pull --ff-only 2>/dev/null || warn "git pull failed for ${target}"
  elif [[ -d "${target}" ]]; then
    warn "Directory ${target} exists but is not a git repo. Skipping clone."
  else
    log "Cloning ${url} -> ${target}"
    git clone "${url}" "${target}" || warn "git clone failed for ${url}"
  fi
}

# log "Creating catkin workspace structure (src + dependencies)..."
mkdir -p src src/arena dependencies/sfm

# Clone core catkin packages into src
clone_or_update "src/tvss_nav" "${LISN_REMOTE}"
clone_or_update "src/tvsn_msgs" "${LISN_MSGS_REMOTE}"
clone_or_update "src/dynamic_obstacle_detector" "${LISN_DOD_REMOTE}"
clone_or_update "src/sfm_local_controller" "${LISN_SFM_REMOTE}"

# Clone lightsfm into a non-catkin dependencies folder
clone_or_update "dependencies/sfm/lightsfm" "${LISN_LIGHTSFM_REMOTE}"

if [[ ${LISN_SKIP_LIGHTSFM_BUILD} -eq 1 ]]; then
  log "Skipping lightsfm build (LISN_SKIP_LIGHTSFM_BUILD=1)"
else
  log "Building and installing lightsfm..."
  pushd dependencies/sfm/lightsfm >/dev/null
  MAKE_JOBS="${LISN_PARALLEL_JOBS}"
  make -j"${MAKE_JOBS}"
  if [[ $(id -u) -eq 0 ]]; then
    make install
  else
    if command -v sudo >/dev/null 2>&1; then
      sudo make install
    else
      warn "sudo not found and not running as root. 'make install' may fail due to write permissions."
      make install
    fi
  fi
  popd >/dev/null
fi

# Arena-Rosnav simulation environment (required for simulation) with Social-Nav replacements
if [[ ${LISN_SKIP_ARENA} -ne 1 ]]; then
  log "Cloning Arena-Rosnav (simulation stack)..."
  clone_or_update "src/arena/arena-rosnav" "${LISN_ARENA_REMOTE}"
  if [[ -n "${LISN_ARENA_COMMIT}" && -d "src/arena/arena-rosnav/.git" ]]; then
    git -C "src/arena/arena-rosnav" checkout "${LISN_ARENA_COMMIT}" || warn "Could not checkout Arena-Rosnav commit ${LISN_ARENA_COMMIT}"
  fi

  if command -v vcs >/dev/null 2>&1 && [[ -f "src/arena/arena-rosnav/.repos" ]]; then
    until vcs import src < src/arena/arena-rosnav/.repos; do
      warn "vcs import failed, retrying in 3s..."
      sleep 3
    done
  else
    warn "vcs (vcstool) not found or .repos missing; skipping vcs import for Arena-Rosnav subrepos."
  fi

  # Replace simulation-setup with Social-Nav version
  # Put it under src/arena/ (alongside other arena packages such as utils)
  if [[ -d "src/arena/arena-rosnav/simulation-setup" ]]; then
    rm -rf "src/arena/arena-rosnav/simulation-setup"
  fi
  if [[ -d "src/arena-rosnav/simulation-setup" ]]; then
    rm -rf "src/arena-rosnav/simulation-setup"
  fi
  if [[ -d "src/arena/simulation-setup" ]]; then
    rm -rf "src/arena/simulation-setup"
  fi
  clone_or_update "src/arena/simulation-setup" "${LISN_SIM_SETUP_REMOTE}"

  # Replace arena evaluation folder with Social-Nav version
  # (Social-Nav/arena-evaluation repo now provides the whole 'evaluation' tree)
  # Put it under src/arena/ (alongside other arena packages such as utils)
  if [[ -d "src/arena/arena-rosnav/arena/evaluation" ]]; then
    rm -rf "src/arena/arena-rosnav/arena/evaluation"
  fi
  if [[ -d "src/arena-rosnav/arena/evaluation" ]]; then
    rm -rf "src/arena-rosnav/arena/evaluation"
  fi
  if [[ -d "src/arena/evaluation" ]]; then
    rm -rf "src/arena/evaluation"
  fi
  clone_or_update "src/arena/evaluation" "${LISN_ARENA_EVAL_REMOTE}"
  if [[ -n "${LISN_ARENA_EVAL_BRANCH}" && -d "src/arena/evaluation/.git" ]]; then
    if [[ ${LISN_SKIP_FETCH} -ne 1 ]]; then
      git -C "src/arena/evaluation" fetch --all --prune >/dev/null 2>&1 || warn "git fetch failed for arena_evaluation"
    fi
    git -C "src/arena/evaluation" checkout "${LISN_ARENA_EVAL_BRANCH}" || warn "Could not checkout arena_evaluation branch ${LISN_ARENA_EVAL_BRANCH}"
    if [[ ${LISN_SKIP_FETCH} -ne 1 ]]; then
      git -C "src/arena/evaluation" pull --ff-only 2>/dev/null || warn "git pull failed for arena_evaluation"
    fi
  fi

  # Remove duplicate upstream packages to avoid name collisions
  # Keep src/arena/{simulation-setup,evaluation} because we intentionally place Social-Nav replacements there.
  for dup in \
    "src/arena_simulation_setup"; do
    if [[ -d "${dup}" ]]; then
      warn "Removing duplicate package at ${dup}"
      rm -rf "${dup}"
    fi
  done
fi

# Install ROS dependencies
if ! command -v rosdep >/dev/null 2>&1; then
  warn "rosdep not found. Skipping dependency installation. Install rosdep and re-run if needed."
else
  # Initialize rosdep (requires root for first-time init)
  if [[ ${LISN_SKIP_ROSDEP} -eq 1 ]]; then
    log "Skipping rosdep install (LISN_SKIP_ROSDEP=1)"
  else
    log "Ensuring rosdep data is available..."
    if [[ $(id -u) -eq 0 ]]; then
      rosdep init || true
      rosdep update --rosdistro "${ROS_DISTRO}" || warn "rosdep update failed; continue and try installing resolvable deps"
    else
      if ! rosdep update --rosdistro "${ROS_DISTRO}" >/dev/null 2>&1; then
        warn "rosdep not initialized for this user. Run: 'sudo rosdep init && rosdep update' as root, or set LISN_SKIP_ROSDEP=1 to skip."
      fi
    fi

    log "Running rosdep to install ROS dependencies..."
    rosdep install --rosdistro "${ROS_DISTRO}" --from-paths src --ignore-src -r -y || warn "rosdep failed; install missing deps manually."
  fi
fi


# Build catkin workspace
if [[ ${LISN_SKIP_BUILD} -eq 1 ]]; then
  log "Skipping catkin build (LISN_SKIP_BUILD=1)"
else
  log "Building catkin workspace..."

  if command -v catkin >/dev/null 2>&1; then
    log "Found 'catkin' (catkin tools). Using 'catkin build'."
    catkin config --source-space src || true
    catkin build --no-status --summarize
  elif command -v catkin_make >/dev/null 2>&1; then
    log "Found 'catkin_make'. Using 'catkin_make'."
    if [[ ! -f src/CMakeLists.txt ]]; then
      log "Initializing src as a catkin workspace..."
      (cd src && catkin_init_workspace)
    fi
    catkin_make
  else
    err "Neither catkin nor catkin_make found; cannot build workspace."
    exit 1
  fi

  if [[ -f "${WS_DIR}/devel/setup.bash" ]]; then
    # Minimal runtime sanity check to catch missing overlays
    source "${WS_DIR}/devel/setup.bash"
    rospack profile >/dev/null 2>&1 || warn "rospack profile failed; check ROS package paths"
    rospack find tvss_nav >/dev/null 2>&1 || warn "tvss_nav package not discoverable after build"
  fi

  log "Build finished."
  echo
  echo "Next steps:"
  echo "  source \"${WS_DIR}/devel/setup.bash\""
  echo "  roslaunch tvss_nav tvss_nav.launch"
  echo
  echo "For Arena-Rosnav simulation: follow README section 1.1 to install Arena-Rosnav, then replace its simulation-setup and arena_evaluation with the Social-Nav forks as described."
fi
