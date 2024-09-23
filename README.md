# ResearchAI


Here’s a simple and clear **README** specifically for managing and using the virtual environment in your project. This will help you remember why and how you set up the virtual environment.

### **README - Virtual Environment for Research AI Project**

---

### Project: **Research AI Project**

#### Purpose:
The virtual environment is used to isolate the project's dependencies (libraries, packages, etc.) from other projects and the system-wide Python installation. By doing this, we ensure that the specific versions of the tools we need are available only for this project and don’t interfere with other Python environments.

We named the virtual environment **`research_ai_env`** to match the project name, making it easy to identify and manage in the future.

---

### Why Use a Virtual Environment?

- **Dependency Isolation**: Prevents conflicts between different projects by keeping project-specific libraries and versions in their own isolated environment.
- **Easy Dependency Management**: Allows you to install the exact libraries required for this project without affecting other Python projects on your system.
- **Portability**: Makes it easier to recreate the same environment on other systems by exporting the list of dependencies (`requirements.txt`).
- **Organized Project Structure**: Naming the virtual environment after the project helps you keep things organized, especially when managing multiple projects.

---

### How to Set Up and Use the Virtual Environment

1. **Create the Virtual Environment** (Only do this once):
   - Run the following command in the terminal within your project directory to create a virtual environment named **`research_ai_env`**:
     ```bash
     rm -rf research_ai_env
     python3 -m venv research_ai_env
     ```

2. **Activate the Virtual Environment**:
   - You need to activate the virtual environment whenever you work on the project.

     ```bash
     source research_ai_env/bin/activate
     ```

   - After activating, your terminal prompt will change to show `(research_ai_env)`, indicating that the virtual environment is active.

3. **Install Project Dependencies**:
   - Once the virtual environment is activated, install the required dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```

4. **Deactivate the Virtual Environment**:
   - When you’re done working, deactivate the virtual environment by simply running:
     ```bash
     deactivate
     ```

---

### How to Reuse the Virtual Environment

Whenever you want to start working on this project in the future:

1. **Navigate to the Project Directory**:
   ```bash
   cd path/to/your/project
   ```

2. **Activate the Virtual Environment**:
   - On **Windows**:
     ```bash
     research_ai_env\Scripts\activate
     ```

   - On **Linux/macOS**:
     ```bash
     source research_ai_env/bin/activate
     ```

3. **Start Working on the Project**:
   You are now inside the project-specific environment where all dependencies are isolated for this project.

---

### Notes:

- Always make sure to activate the virtual environment before running your project’s scripts or installing new packages.
- The **`research_ai_env`** directory contains all the files related to the virtual environment. This folder should **not be shared** when you distribute the project. Only the `requirements.txt` file, which lists your dependencies, needs to be shared.
- To recreate the environment on another machine, run:
  ```bash
  pip install -r requirements.txt
  ```

---




# Ollama Service Setup and Troubleshooting Guide

## 1. Checking Ollama Service Status
To check if the Ollama service is running, use the following command:
```
sudo systemctl status ollama.service
```
If the service is not running, follow the steps below to set it up.

## 2. Creating a Systemd Service for Ollama

1. Create a systemd service file for Ollama:
   ```bash
   sudo nano /etc/systemd/system/ollama.service
   ```

2. Add the following content (update the paths as needed):
   ```ini
   [Unit]
   Description=Ollama Server
   After=network.target

   [Service]
   ExecStart=/path/to/ollama/binary --port 11434
   WorkingDirectory=/path/to/ollama/
   Restart=always
   User=your-username
   Group=your-group
   Environment=DISPLAY=:0
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```

3. Reload systemd to recognize the new service:
   ```bash
   sudo systemctl daemon-reload
   ```

4. Enable the Ollama service to start on boot:
   ```bash
   sudo systemctl enable ollama.service
   ```

5. Start the service:
   ```bash
   sudo systemctl start ollama.service
   ```

6. Check the status:
   ```bash
   sudo systemctl status ollama.service
   ```

## 3. Testing the Ollama Model (Llama2-uncensored)

To test if the API and the `llama2-uncensored` model are working correctly, use the following `curl` command:
```bash
curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d '{"model": "llama2-uncensored", "prompt": "What is the capital of France?", "max_tokens": 100}'
```

You should receive a JSON response if everything is working.

## 4. Checking Logs for Errors
If the model doesn't work as expected, check the logs:
```bash
sudo journalctl -u ollama.service
```

## 5. Updating the Systemd Service File
If you see warnings about `syslog` being obsolete, update the service file:
1. Open the file for editing:
   ```bash
   sudo nano /etc/systemd/system/ollama.service
   ```
2. Replace:
   ```ini
   StandardOutput=syslog
   StandardError=syslog
   ```
   with:
   ```ini
   StandardOutput=journal
   StandardError=journal
   ```
3. Save, then reload and restart the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama.service
   ```

## Additional Notes
- Make sure your Ollama server is listening on the correct port (default: 11434).
- Check GPU compatibility if you are using GPU inference.

## Date of Last Update: 2024-09-23
