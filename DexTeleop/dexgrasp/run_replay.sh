#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${GREEN}=======================================================${NC}"
echo -e "${GREEN}             Allegro Hand Trajectory Replayer           ${NC}"
echo -e "${GREEN}=======================================================${NC}"

# Default trajectory file
DEFAULT_TRAJECTORY="recorded_trajectories/trajectory_20250429-190223.json"

# Check if a trajectory file was provided as an argument
if [ "$1" != "" ]; then
    TRAJECTORY="$1"
    echo -e "${YELLOW}Using specified trajectory file: ${BLUE}$TRAJECTORY${NC}"
else
    TRAJECTORY="$DEFAULT_TRAJECTORY"
    echo -e "${YELLOW}Using default trajectory file: ${BLUE}$TRAJECTORY${NC}"
fi

# Check if the file exists
if [ ! -f "$TRAJECTORY" ]; then
    echo -e "${YELLOW}WARNING: Trajectory file not found at ${BLUE}$TRAJECTORY${NC}"
    echo -e "${YELLOW}Will attempt to use the path as specified (might be relative to script location)${NC}"
fi

echo -e "${GREEN}Starting trajectory replay...${NC}"
echo -e "${YELLOW}Controls:${NC}"
echo -e "${YELLOW}  ESC: Quit the program${NC}"
echo -e "${YELLOW}  V: Toggle viewer sync${NC}"
echo -e "${YELLOW}  R: Start replaying${NC}"
echo -e "${YELLOW}  T: Stop replaying${NC}"
echo -e "${GREEN}=======================================================${NC}"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the replay script
python "${SCRIPT_DIR}/tasks/replay_trajectory.py" --trajectory "$TRAJECTORY"

# Usage: ./run_replay.sh [path/to/trajectory.json] 