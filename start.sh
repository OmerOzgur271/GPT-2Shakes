
#!/bin/bash

# Start the FastAPI server in a background process
python myapp.py &

# Print a message to indicate that the server has started
echo "Server is running..."

# Wait for the server to finish
wait
