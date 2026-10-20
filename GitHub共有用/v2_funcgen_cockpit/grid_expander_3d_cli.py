"""Preview a concentric 3D grid; execute only after explicit human confirmation."""
import argparse
from decimal import Decimal
import itertools
import json
import math
from pathlib import Path
import sys

from core.database import Database3DManager
from core.xfoil_runner import analyze_funcgen_airfoil_all_re
from core.paths import DEFAULT_DATABASE, DEFAULT_GRID_CONFIG, DATABASE_DIR

BASE_DIR = Path(__file__).resolve().parent


def sequence(start, stop, step, minimum, maximum):
    values = (start, stop, step)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('格子の範囲と刻みは有限の数値にしてください')
    if not minimum <= start <= stop <= maximum or step <= 0:
        raise ValueError('格子の範囲または刻みが不正です')
    a, b, delta = map(lambda v: Decimal(str(v)), values)
    count = int((b-a)//delta)+1
    if count > 100000:
        raise ValueError('格子点数が過大です')
    return [float(a+i*delta) for i in range(count)]


def plan_grid(config, completed=()):
    tc = sequence(config['tc_min'], config['tc_max'], config['tc_step'], 20., 30.)
    xt = sequence(config['xt_min'], config['xt_max'], config['xt_step'], 38., 48.)
    if any(round(v, 1) != v for v in tc+xt):
        raise ValueError('DB完了キーに合わせ、tc/xtは0.1%単位にしてください')
    kle = config['kle_list']
    if (not isinstance(kle, list) or not kle or any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or not .6 <= v <= 1.5 or round(v, 2) != v for v in kle)):
        raise ValueError('kle_listは0.6〜1.5、0.01単位の数値リストにしてください')
    kle = sorted(set(kle))
    if len(tc)*len(xt)*len(kle) > 1000000:
        raise ValueError('格子が大きすぎます。範囲を分割してください')
    center = [config['center_tc'], config['center_xt'], config['center_kle']]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in center):
        raise ValueError('中心点は有限の数値にしてください')
    if not (20 <= center[0] <= 30 and 38 <= center[1] <= 48 and .6 <= center[2] <= 1.5):
        raise ValueError('中心点が設計範囲外です')
    scales = [config['tc_step'], config['xt_step'], min((b-a for a,b in zip(kle, kle[1:])), default=1.)]
    completed = set(completed)
    candidates = [point for point in itertools.product(tc, xt, kle)
                  if (round(point[0],1), round(point[1],1), round(point[2],2)) not in completed]
    # Dimensionless Chebyshev shells give one grid step comparable weight per axis.
    candidates.sort(key=lambda p: (max(abs(v-c)/s for v,c,s in zip(p,center,scales)), p))
    return candidates, len(tc)*len(xt)*len(kle)


def analysis_ranges(config):
    reynolds = sequence(config['reynolds_min'], config['reynolds_max'], config['reynolds_step'], 1, 1e9)
    if any(not v.is_integer() for v in reynolds):
        raise ValueError('Reynolds数は整数で指定してください')
    alphas = sequence(config['alpha_min'], config['alpha_max'], config['alpha_step'], 0., 12.)
    return [int(v) for v in reynolds], alphas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_GRID_CONFIG)
    parser.add_argument('--csv', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--method', choices=['warp', 'cst'], default='warp')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--run', action='store_true', help='計画表示後、承認入力を受けて計算開始')
    parser.add_argument('--dry-run', action='store_true', help='計画表示のみ（既定）')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 64:
        parser.error('--workersは1〜64です')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    db = Database3DManager(csv_path=args.csv)
    if args.csv.exists():
        # Preview is read-only: no cache writes or implicit legacy migration.
        import pandas as pd
        db.df = db._normalize(pd.read_csv(args.csv))
    points, total = plan_grid(config, db.get_completed_keys(args.method))
    reynolds, alphas = analysis_ranges(config)
    cases = len(points)*len(reynolds)
    print(f'方式: {args.method}; 全形状: {total}; 未完了: {len(points)}; Re点数: {len(reynolds)}')
    print(f'1形状あたり迎角: {len(alphas)}点; XFOILケース数: {cases}; 並列数: {args.workers}')
    print(f'範囲: tc={config["tc_min"]}..{config["tc_max"]} step={config["tc_step"]}; '
          f'xt={config["xt_min"]}..{config["xt_max"]} step={config["xt_step"]}; kle={config["kle_list"]}')
    timeout_hours = len(points)*math.ceil(len(reynolds)/args.workers)*30/3600
    print(f'時間目安: 全Reが30秒タイムアウトした場合の待機予算 {timeout_hours:.2f}時間 + 生成/保存時間。実時間は未測定。')
    print(f'先頭候補: {points[:5]}')
    if args.dry_run or not args.run or not points:
        print('ドライラン終了。計算・DB変更は行っていません。')
        return 0
    print('上記パラメータ範囲と時間目安を確認した人だけ開始してください。')
    if input('実行を承認する場合は START と入力: ').strip() != 'START':
        print('開始しませんでした。')
        return 0
    pending = []
    try:
        for index, (tc, xt, kle) in enumerate(points, 1):
            print(f'形状 {index}/{len(points)}: {tc}, {xt}, {kle}')
            analyze_funcgen_airfoil_all_re(tc, xt, kle, reynolds,
                alpha_min=config['alpha_min'], alpha_max=config['alpha_max'], alpha_step=config['alpha_step'],
                max_workers=args.workers, airfoils_dir=DATABASE_DIR / 'airfoils', generator_method=args.method,
                records_callback=pending.extend,
                progress_callback=lambda done, count: print(f'  Re完了 {done}/{count}'))
            db.append_records(pending)
            pending.clear()
    except KeyboardInterrupt:
        print('\n中断要求。実行中のケース終了後、完了データを保存します。')
        return 130
    finally:
        if pending:
            print(f'完了レコードを保存: {db.append_records(pending)}件')
    return 0


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, RuntimeError, OSError) as error:
        print(f'エラー: {error}')
        raise SystemExit(1)
    except (KeyboardInterrupt, EOFError):
        print('\n開始せず終了しました。')
        raise SystemExit(130)
