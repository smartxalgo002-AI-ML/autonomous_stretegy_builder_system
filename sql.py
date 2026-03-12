import sqlite3
conn = sqlite3.connect("trading_system.db")

for row in conn.execute("SELECT * FROM backtest_results LIMIT 5"):
    print(row)