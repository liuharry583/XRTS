# 数据库目录

- `db_v1/`：原先散落在 database 根目录的 1887 份 NPZ、manifest.csv、qc_report.csv。
- `db_v2/`：原先 database/db_v2/db_v2 的同名文件，已提升到第一层。
- `organization_manifest.json`：3778 个文件移动前的路径与 SHA-256；移动后逐一校验一致。

两个版本各有 1700 个 done 算例、187 个 zero_fill 算例；manifest 中另有 291 个 skipped_singular 记录，没有对应谱文件。
`data_reading.py` 默认路径已调整为 database/db_v1。

db_v2 中 E_eV 是光子能量位移，下移为负；数组随 omega 升序存储，因此读取时必须将 E_eV 与对应谱值一起排序，不能只翻转一个数组。

db_v2 为 Be、E0=5.0 keV、Zf=2、无碰撞、Arkhipov、弹性宽度 2 eV，method=auto。
该库不是 Dornheim 2.96 keV 的同条件数据库，不应直接将拟合结果解释为该实验的可靠温度/密度诊断。
