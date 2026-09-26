# Recorded football-data.co.uk fixtures

Downloaded on 2026-09-26 from `https://football-data.co.uk/mmz4281/{season_code}/{code}.csv`:

| File           | Competition | Season  | Notes                                        |
| -------------- | ----------- | ------- | -------------------------------------------- |
| `E0_1415.csv`  | EPL         | 2014-15 | Old layout: no `Time` column; trailing empty row |
| `E0_2425.csv`  | EPL         | 2024-25 | Current layout: UTF-8 BOM, `Time` column      |
| `SP1_1415.csv` | LALIGA      | 2014-15 | Old layout, no `Referee` column either        |
| `SP1_2425.csv` | LALIGA      | 2024-25 | Current layout, UTF-8 BOM                     |
| `E0_corrupt.csv` | EPL       | 2024-25 | Hand-broken copy for the validation tests: negative goals, a duplicated fixture, a 1.00 price |

Bytes are kept exactly as served (CRLF line endings, BOM): the checksum tests depend on it,
so `.gitattributes` marks this directory as binary. Tests never re-download them.
