@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo ABLATION TEST SETUP
echo ============================================================

set /p LOW_PORT=Enter low port number:
set /p NUM_PORTS=Enter number of ports:
set /p START_RUN=Start from run number (1-21, default 1):
if "!START_RUN!"=="" set START_RUN=1

set BASE_CMD=python evaluate_qald9.py --input-file qald_9_plus_test_dbpedia.json --api-url http://localhost:8000/api --sparql-endpoint https://dbpedia.org/sparql --lang en --model-name openai/gpt-4o-mini --cost-prompt 0.15 --cost-completion 0.60 --log-calls --num-ports %NUM_PORTS% --low-port %LOW_PORT%

echo.
echo ============================================================
echo ABLATION TEST - 7 conditions x 3 runs = 21 total
echo Ports: %LOW_PORT% - %LOW_PORT% + %NUM_PORTS% - 1
echo Starting from run: %START_RUN%
echo ============================================================

:: ============================================================
:: Prompt upfront for directories of already-completed runs
:: (kept outside the loop to avoid batch parser issues with
::  parentheses inside nested blocks)
:: ============================================================
if %START_RUN% GTR 1 (
    echo.
    echo Enter result folder names for already-completed runs.
    echo Folder names are inside test_results\.
    echo.
)

if %START_RUN% GTR 1  set /p DIR_eat_1=  Run  1 of 21 - no-eat run1:
if %START_RUN% GTR 2  set /p DIR_context_1=  Run  2 of 21 - no-context run1:
if %START_RUN% GTR 3  set /p DIR_icl_1=  Run  3 of 21 - no-icl run1:
if %START_RUN% GTR 4  set /p DIR_context_icl_1=  Run  4 of 21 - no-context+no-icl run1:
if %START_RUN% GTR 5  set /p DIR_icl_eat_1=  Run  5 of 21 - no-icl+no-eat run1:
if %START_RUN% GTR 6  set /p DIR_context_eat_1=  Run  6 of 21 - no-context+no-eat run1:
if %START_RUN% GTR 7  set /p DIR_all_1=  Run  7 of 21 - no-context+no-eat+no-icl run1:
if %START_RUN% GTR 8  set /p DIR_eat_2=  Run  8 of 21 - no-eat run2:
if %START_RUN% GTR 9  set /p DIR_context_2=  Run  9 of 21 - no-context run2:
if %START_RUN% GTR 10 set /p DIR_icl_2=  Run 10 of 21 - no-icl run2:
if %START_RUN% GTR 11 set /p DIR_context_icl_2=  Run 11 of 21 - no-context+no-icl run2:
if %START_RUN% GTR 12 set /p DIR_icl_eat_2=  Run 12 of 21 - no-icl+no-eat run2:
if %START_RUN% GTR 13 set /p DIR_context_eat_2=  Run 13 of 21 - no-context+no-eat run2:
if %START_RUN% GTR 14 set /p DIR_all_2=  Run 14 of 21 - no-context+no-eat+no-icl run2:
if %START_RUN% GTR 15 set /p DIR_eat_3=  Run 15 of 21 - no-eat run3:
if %START_RUN% GTR 16 set /p DIR_context_3=  Run 16 of 21 - no-context run3:
if %START_RUN% GTR 17 set /p DIR_icl_3=  Run 17 of 21 - no-icl run3:
if %START_RUN% GTR 18 set /p DIR_context_icl_3=  Run 18 of 21 - no-context+no-icl run3:
if %START_RUN% GTR 19 set /p DIR_icl_eat_3=  Run 19 of 21 - no-icl+no-eat run3:
if %START_RUN% GTR 20 set /p DIR_context_eat_3=  Run 20 of 21 - no-context+no-eat run3:

set RUN_COUNT=0
set TOTAL_RUNS=21

for /l %%R in (1,1,3) do (
    echo.
    echo ############################################################
    echo # RUN %%R / 3
    echo ############################################################

    echo.
    set /a RUN_COUNT+=1
    echo --- [!RUN_COUNT!/%TOTAL_RUNS%] Run %%R - --no-eat ---
    if !RUN_COUNT! GEQ !START_RUN! (
        %BASE_CMD% --test-name ABLATION_no-eat_run%%R --no-eat
        for /f "delims=" %%d in ('dir /b /od test_results 2^>nul') do set LAST_DIR=%%d
        set DIR_eat_%%R=!LAST_DIR!
        python -c "import json; d=json.load(open('test_results/!LAST_DIR!/summary.json')); print('  --> Cost: $'+f'{d[\"total_cost_so_far\"]:.4f}')"
    ) else (
        echo   [SKIPPED]
    )

    echo.
    set /a RUN_COUNT+=1
    echo --- [!RUN_COUNT!/%TOTAL_RUNS%] Run %%R - --no-context ---
    if !RUN_COUNT! GEQ !START_RUN! (
        %BASE_CMD% --test-name ABLATION_no-context_run%%R --no-context
        for /f "delims=" %%d in ('dir /b /od test_results 2^>nul') do set LAST_DIR=%%d
        set DIR_context_%%R=!LAST_DIR!
        python -c "import json; d=json.load(open('test_results/!LAST_DIR!/summary.json')); print('  --> Cost: $'+f'{d[\"total_cost_so_far\"]:.4f}')"
    ) else (
        echo   [SKIPPED]
    )

    echo.
    set /a RUN_COUNT+=1
    echo --- [!RUN_COUNT!/%TOTAL_RUNS%] Run %%R - --no-icl ---
    if !RUN_COUNT! GEQ !START_RUN! (
        %BASE_CMD% --test-name ABLATION_no-icl_run%%R --no-icl
        for /f "delims=" %%d in ('dir /b /od test_results 2^>nul') do set LAST_DIR=%%d
        set DIR_icl_%%R=!LAST_DIR!
        python -c "import json; d=json.load(open('test_results/!LAST_DIR!/summary.json')); print('  --> Cost: $'+f'{d[\"total_cost_so_far\"]:.4f}')"
    ) else (
        echo   [SKIPPED]
    )

    echo.
    set /a RUN_COUNT+=1
    echo --- [!RUN_COUNT!/%TOTAL_RUNS%] Run %%R - --no-context --no-icl ---
    if !RUN_COUNT! GEQ !START_RUN! (
        %BASE_CMD% --test-name ABLATION_no-context_no-icl_run%%R --no-context --no-icl
        for /f "delims=" %%d in ('dir /b /od test_results 2^>nul') do set LAST_DIR=%%d
        set DIR_context_icl_%%R=!LAST_DIR!
        python -c "import json; d=json.load(open('test_results/!LAST_DIR!/summary.json')); print('  --> Cost: $'+f'{d[\"total_cost_so_far\"]:.4f}')"
    ) else (
        echo   [SKIPPED]
    )

    echo.
    set /a RUN_COUNT+=1
    echo --- [!RUN_COUNT!/%TOTAL_RUNS%] Run %%R - --no-icl --no-eat ---
    if !RUN_COUNT! GEQ !START_RUN! (
        %BASE_CMD% --test-name ABLATION_no-icl_no-eat_run%%R --no-icl --no-eat
        for /f "delims=" %%d in ('dir /b /od test_results 2^>nul') do set LAST_DIR=%%d
        set DIR_icl_eat_%%R=!LAST_DIR!
        python -c "import json; d=json.load(open('test_results/!LAST_DIR!/summary.json')); print('  --> Cost: $'+f'{d[\"total_cost_so_far\"]:.4f}')"
    ) else (
        echo   [SKIPPED]
    )

    echo.
    set /a RUN_COUNT+=1
    echo --- [!RUN_COUNT!/%TOTAL_RUNS%] Run %%R - --no-context --no-eat ---
    if !RUN_COUNT! GEQ !START_RUN! (
        %BASE_CMD% --test-name ABLATION_no-context_no-eat_run%%R --no-context --no-eat
        for /f "delims=" %%d in ('dir /b /od test_results 2^>nul') do set LAST_DIR=%%d
        set DIR_context_eat_%%R=!LAST_DIR!
        python -c "import json; d=json.load(open('test_results/!LAST_DIR!/summary.json')); print('  --> Cost: $'+f'{d[\"total_cost_so_far\"]:.4f}')"
    ) else (
        echo   [SKIPPED]
    )

    echo.
    set /a RUN_COUNT+=1
    echo --- [!RUN_COUNT!/%TOTAL_RUNS%] Run %%R - --no-context --no-eat --no-icl ---
    if !RUN_COUNT! GEQ !START_RUN! (
        %BASE_CMD% --test-name ABLATION_no-context_no-eat_no-icl_run%%R --no-context --no-eat --no-icl
        for /f "delims=" %%d in ('dir /b /od test_results 2^>nul') do set LAST_DIR=%%d
        set DIR_all_%%R=!LAST_DIR!
        python -c "import json; d=json.load(open('test_results/!LAST_DIR!/summary.json')); print('  --> Cost: $'+f'{d[\"total_cost_so_far\"]:.4f}')"
    ) else (
        echo   [SKIPPED]
    )
)

echo.
echo ============================================================
echo ALL 21 ABLATION RUNS COMPLETE - GENERATING F1 SUMMARY
echo ============================================================

python -c ^
"import json, os^

^

def f1(d): return d.get('macro_f1_so_far', 0.0)^

^

conditions = [^
    ('no-eat',              ['%DIR_eat_1%',         '%DIR_eat_2%',         '%DIR_eat_3%']),^
    ('no-context',          ['%DIR_context_1%',     '%DIR_context_2%',     '%DIR_context_3%']),^
    ('no-icl',              ['%DIR_icl_1%',         '%DIR_icl_2%',         '%DIR_icl_3%']),^
    ('no-context+no-icl',   ['%DIR_context_icl_1%', '%DIR_context_icl_2%', '%DIR_context_icl_3%']),^
    ('no-icl+no-eat',       ['%DIR_icl_eat_1%',     '%DIR_icl_eat_2%',     '%DIR_icl_eat_3%']),^
    ('no-context+no-eat',   ['%DIR_context_eat_1%', '%DIR_context_eat_2%', '%DIR_context_eat_3%']),^
    ('no-context+no-eat+no-icl', ['%DIR_all_1%',   '%DIR_all_2%',         '%DIR_all_3%']),^
]^

^

print()^
print('=' * 72)^
print(f\"{'Condition':<30} {'Run1':>8} {'Run2':>8} {'Run3':>8} {'Avg':>8}\")^
print('=' * 72)^

all_avgs = []^
for name, dirs in conditions:^
    scores = []^
    for d in dirs:^
        path = os.path.join('test_results', d, 'summary.json')^
        try:^
            data = json.load(open(path))^
            scores.append(f1(data))^
        except Exception as e:^
            scores.append(float('nan'))^
    avg = sum(s for s in scores if s == s) / max(1, sum(1 for s in scores if s == s))^
    all_avgs.append(avg)^
    row = f'{name:<30} ' + ' '.join(f'{s:>8.4f}' for s in scores) + f' {avg:>8.4f}'^
    print(row)^

print('=' * 72)^
overall = sum(all_avgs) / len(all_avgs)^
print(f\"{'OVERALL AVERAGE':<30} {'':>8} {'':>8} {'':>8} {overall:>8.4f}\")^
print('=' * 72)^
"

echo.
pause
