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

This README will help guide you in using and managing the virtual environment for this project in the future.