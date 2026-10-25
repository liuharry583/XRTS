# figure.png 数字化过程产物

本文件夹集中保存提取 figure.png、坐标标定、镜像与平滑过程中产生的文件。

- calibration_*：标定放大图、刻度实测和配置差异。
- compare_*：提取描点与平滑对比图。
- exp_0.txt / sif_0.txt：镜像后的原始提取数据。
- exp_data.txt / sif_data.txt：最终平滑数据的归档副本。上层同名文件是现有 RMS 脚本实际读取的工作输入，两份当前完全一致；以后重新提取不会自动同步归档副本。
- extraction_run.log / smoothing_run.log：运行日志。
- mirror_*：镜像重跑的报告和统计，描述当前成品。
- digitization_*：首次未镜像运行的历史报告和统计，不代表当前成品。
- before_mirror_data.npz：镜像前数据快照。
- distill_data.py.before_calibration：最初提取脚本备份。
- archive_manifest.json：归档文件及 SHA-256 校验。

原始 figure.png、正在使用的两份最终数据及 Python 脚本保留在上层；没有更改脚本路径或运行逻辑。报告中原来的过程产物路径属于历史记录，归档后的实际位置以本目录为准。
