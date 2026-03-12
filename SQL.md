Your **`trading_system.db`** is an SQLite database. Here are the simplest ways to **see the data inside it**.

---

# 1️⃣ See All Tables in the Database

Run in your terminal:

```bash
python -c "import sqlite3; conn=sqlite3.connect('trading_system.db'); print(conn.execute('SELECT name FROM sqlite_master WHERE type=\"table\"').fetchall()); conn.close()"
```

Example output:

```
[('strategies',), ('backtest_results',), ('evaluations',)]
```

---

# 2️⃣ View Data in a Table

Example for **backtest results**:

```bash
python -c "import sqlite3; conn=sqlite3.connect('trading_system.db'); print(conn.execute('SELECT * FROM backtest_results LIMIT 10').fetchall()); conn.close()"
```

This prints the **first 10 rows**.

---

# 3️⃣ See Table Structure (Columns)

Example:

```bash
python -c "import sqlite3; conn=sqlite3.connect('trading_system.db'); print(conn.execute('PRAGMA table_info(backtest_results)').fetchall()); conn.close()"
```

Output will show columns like:

```
[(0, 'strategy_id', 'TEXT', 0, None, 0),
 (1, 'win_rate', 'REAL', 0, None, 0),
 (2, 'profit_factor', 'REAL', 0, None, 0)]
```

---

# 4️⃣ See Strategies Stored

```bash
python -c "import sqlite3; conn=sqlite3.connect('trading_system.db'); print(conn.execute('SELECT strategy_id, name FROM strategies LIMIT 10').fetchall()); conn.close()"
```

---

# 5️⃣ See Best Strategies

Example:

```bash
python -c "import sqlite3; conn=sqlite3.connect('trading_system.db'); print(conn.execute('SELECT strategy_id, win_rate, profit_factor FROM backtest_results ORDER BY profit_factor DESC LIMIT 10').fetchall()); conn.close()"
```

---

# 6️⃣ Best Way (Visual Viewer)

Install a GUI tool:

### DB Browser for SQLite

Download:

```
https://sqlitebrowser.org
```

Then:

1. Open the program
2. Click **Open Database**
3. Select

```
trading_system.db
```

Now you can **browse tables visually**.

---

# 7️⃣ Quick Interactive Mode (Python)

Run:

```bash
python
```

Then:

```python
import sqlite3
conn = sqlite3.connect("trading_system.db")

for row in conn.execute("SELECT * FROM backtest_results LIMIT 5"):
    print(row)
```

---

✅ **Easiest command to explore everything**

```bash
python -c "import sqlite3; conn=sqlite3.connect('trading_system.db'); print(conn.execute('SELECT name FROM sqlite_master WHERE type=\"table\"').fetchall()); conn.close()"
```

This shows **all tables stored by your AI trading system**.
