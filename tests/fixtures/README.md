# Test fixtures

Hand-written, tiny versions of the dbt marts with the same columns. They let tests
run without BigQuery (CI has no cloud access) and cover edge cases on purpose:

| User | Case |
|---|---|
| u1 | Single touchpoint, no purchase |
| u2 | Three channels, converts on the last touch |
| u3 | Unknown channel (no medium), reached the cart |
| u4 | Obfuscated channel, then converts via Direct |
| u5 | Same channel twice, no purchase |
