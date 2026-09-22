# How to run the experiment (Windows) - step by step

Follow the steps in order. Every grey box is one command: copy it, paste it, press **Enter**, and wait until it finishes before the next one.

---

## Step 1 - Unzip the kit

1. Right-click `casp_experiment_kit.zip` and choose **Extract All...**, then click **Extract**.
2. Open the extracted folder, then open the folder named **casp_experiment** inside it.
3. You are in the right folder when you can see the files `run_experiment.py`, `analyze.py` and `README.md`.

---

## Step 2 - Open PowerShell inside that folder

While you are inside the **casp_experiment** folder in File Explorer:

1. Click once on the **address bar** at the top of the window (where the folder path is shown).
2. Delete the text, type `powershell`, and press **Enter**.

A blue or black window opens. This is where you paste every command below.
Its first line should end with `\casp_experiment>`, which means it is already in the right folder.

**How to paste:** press **Ctrl + V** or right-click inside the window, then press **Enter**.

---

## Step 3 - Check that Python is installed

```
python --version
```

- If you see something like `Python 3.12.x`, go to Step 4.
- If you see an error: install Python from https://www.python.org/downloads/ and tick the box **"Add python.exe to PATH"** during installation. Then close PowerShell and repeat Step 2.

---

## Step 4 - Install the required libraries (only once)

```
pip install -r requirements.txt
```

---

## Step 5 - Enter your API key

Copy this command, replace `PASTE-YOUR-KEY-HERE` with your key (it starts with `sk-ant-`), keep the quotation marks, then press Enter:

```
$env:ANTHROPIC_API_KEY="PASTE-YOUR-KEY-HERE"
```

Check that the key was saved (it should print your key):

```
echo $env:ANTHROPIC_API_KEY
```

**Important:** the key is remembered only in this window. If you close PowerShell, repeat Step 2 and Step 5.

---

## Step 6 - Quick test (costs less than 1 cent)

```
python run_experiment.py --probe
```

- **Good result:** lines starting with `OK`.
- **Lines starting with `ERROR`:** stop here and send the output to Claude.

---

## Step 7 - Small trial run (about 1-2 minutes)

```
python run_experiment.py --pilot
```

---

## Step 8 - Full experiment (may take 1-2 hours)

```
python run_experiment.py
```

- You will see progress lines such as `progress 300/5100`.
- If it stops (internet cut, window closed), repeat Step 2 and Step 5, then run the same command again. It continues from where it stopped; nothing is lost.

---

## Step 9 - Produce the tables and figures

```
python analyze.py --results results
```

---

## Step 10 - Package the results to send

```
Compress-Archive -Path results -DestinationPath results.zip -Force
```

A file named **results.zip** appears in the casp_experiment folder. Upload it to Claude in the chat.

---

### Estimated cost

About 15-20 US dollars in total for both models (September 2026 prices).

### Never share your API key

Do not paste the key into the chat or send it in any file. It stays only on your computer.
