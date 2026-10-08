import yfinance as yf

df = yf.download("AAPL", start="2015-01-01", end="2026-01-01")

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

df = df.dropna()

print(df.head())
print("\n\n", df.iloc[0])
print("\n\n", df.iloc[0,0])
print("\nnumber of 1's and 0's:", (df["Target"]).value_counts())