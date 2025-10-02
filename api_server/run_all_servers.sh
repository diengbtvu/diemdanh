#!/bin/bash

echo "========================================"
echo "Starting All Face Recognition API Servers"
echo "========================================"
echo ""
echo "Starting servers on:"
echo "  - Vanilla CNN:   http://localhost:5001"
echo "  - ResNet50:      http://localhost:5002"
echo "  - Attention CNN: http://localhost:5003 (RECOMMENDED)"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "========================================"
echo ""

# Start servers in background
python server_vanilla_cnn.py &
PID1=$!

sleep 2

python server_resnet50.py &
PID2=$!

sleep 2

python server_attention_cnn.py &
PID3=$!

echo ""
echo "All servers started!"
echo "PIDs: $PID1, $PID2, $PID3"
echo ""
echo "Check Swagger docs at:"
echo "  - http://localhost:5001/swagger/"
echo "  - http://localhost:5002/swagger/"
echo "  - http://localhost:5003/swagger/"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for Ctrl+C
trap "echo 'Stopping all servers...'; kill $PID1 $PID2 $PID3; exit" INT
wait

