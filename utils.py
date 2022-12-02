from copy import deepcopy
import io
import sys
from time import sleep
from time import time
from typing import Tuple, List, Set
import pandas as pd
import os
from tqdm import tqdm
import sqlite3

# 单元格类型
UNCHECK = -1
ERROR = 0
TRUE = 1
# 规则类型
RULE_JUST_TRUE = 0 # X -> t0._xx = 1，对应 SQL WHERE X
RULE_JUST_ERROR = 1 # X -> t0._xx = 0，对应 SQL WHERE X
RULE_ALL_SAME = 2 # X -> t0._xx = t1._xx，对应 SQL WHERE X
RULE_NORMAL_SINGLE_JUST_TRUE = 3 # X -> t0.a=123，对应 SQL WHERE X AND t0.a=123
RULE_NORMAL_SINGLE_JUST_ERROR = 4 # X -> t0.a=123，对应 SQL WHERE X AND t0.a<>123
RULE_NORMAL_MULTI_OPPOSITE = 5 # X -> t0.a=t1.a。对应 SQL WHERE X AND t0.a<>t1.a
RULE_NORMAL_MULTI_JUST_T1_TRUE = 6 # X -> t1.a=123，对应 SQL WHERE X AND t1.a=123
RULE_NORMAL_MULTI_JUST_T1_ERROR = 7 # X -> t1.a=123，对应 SQL WHERE X AND t1.a<>123
RULE_REGEX = 8 # 正则规则 tab(t0) ^ regular( t0.countyname , 're_str') -> true

SQLITE3_TEMP_FILE = 'sqlite3.tmp'
TEMP_TABLE_NAME = 'tab'
AUTO_INCRE_COL = "aic"


'''加载 csv 将空串改为 null 以适应 RDS 的规则，同时列名全部小写以适应 ROCK 现状
'''
def load_csv(filename:str)->pd.DataFrame:
    import re
    def try_extract_data(dataStr:str)->Tuple[bool, str]:
        # case1: 2/15/04 1/13/12
        match = re.match(r'^(\d{1,2})\/(\d{1,2})\/(\d{2})$', dataStr)
        if match is not None:
            m = int(match.group(1))
            d = int(match.group(2))
            y = int(match.group(3))
            if y < 20:
                y = y + 2000
            else:
                y = y + 1900
            
            return True, f"{y}-{m}-{d}"

        # case2: 2001/6/12
        match = re.match(r'^(\d{4})\/(\d{1,2})\/(\d{1,2})$', dataStr)
        if match is not None:
            y = int(match.group(1))
            m = int(match.group(2))
            d = int(match.group(3))
            return True, f"{y}-{m}-{d}"
        
        return False, None

    tab = pd.read_csv(filename, dtype=str, keep_default_na = False)
    # 空串改为 null
    for i in range(len(tab)):
        for c in tab.columns:
            if len(tab.loc[i, c]) == 0:
                tab.loc[i, c] = 'null'
            # isData, data = try_extract_data(str(tab.loc[i, c]))
            # if isData:
            #     tab.loc[i, c] = data
                
    # 字段都小写
    tab.rename(columns=str.lower, inplace=True)
    tab.insert(column=AUTO_INCRE_COL, loc=0, value=range(len(tab)))
    return tab

'''增加判别列，以 _ 开头，默认值为 UNCHECK
'''
def append_judge_column(table:pd.DataFrame, id_column:str)->pd.DataFrame:
    table = deepcopy(table)
    for column in table.columns:
        if column == id_column:
            continue
        table.insert(column = '_' + column, value = UNCHECK, loc = len(table.columns))
    return table


'''利用 clean 表生成标准解
'''
def label_by_clean(label_dirty_table:pd.DataFrame, clean_table:pd.DataFrame, id_column:str)->pd.DataFrame:
    label_dirty_table = deepcopy(label_dirty_table)
    for i in range(len(label_dirty_table)):
        for c in clean_table.columns:
            if c == id_column:
                continue
            if clean_table.loc[i, c] == label_dirty_table.loc[i, c]:
                label_dirty_table.loc[i, '_' + c] = TRUE
            else:
                label_dirty_table.loc[i, '_' + c] = ERROR
    return label_dirty_table


'''验证并输出标注结果
@param start 验证开始行
@param end 验证结束行，不包括自身，None 值表示 len(table)
'''
def check_result(label_dirty_table:pd.DataFrame, validate_dirty_tab:pd.DataFrame, clean_table:pd.DataFrame, id_column:str, start:int=0, end:int=None, verbose:bool=False):
    # 所有需要判断的单元格数量
    cell_num = 0
    # 未查验。实际正确，错误
    unchecked, unchecked_True, unchecked_Error = 0,0,0
    # 查出正确。查出正确实际正确，错误
    checkTrue, checkTrue_True, checkTrue_Error = 0,0,0
    # 查出错误。查出错误实际正确，错误
    checkError, checkError_True, checkError_Error = 0,0,0
    # 正确查出错误
    good = 0
    # 假阳性。判断为 0，实际 1
    false_positive = 0
    # 假阴性。漏判。判断为 1/-1，实际 0
    false_negative = 0
    # 实际错误数量
    exact_error = 0

    if end is None:
        end = len(label_dirty_table)
    for i in range(start, end):
        for c in label_dirty_table.columns:
            if not c.startswith('_') or c == id_column:
                continue
            cell_num += 1
            judge = label_dirty_table.loc[i, c]
            exact = validate_dirty_tab.loc[i, c]
            if judge == UNCHECK :
                unchecked +=1
                judge = TRUE # 视为 1 正确解
                if exact == TRUE:
                    unchecked_True += 1
                else:
                    unchecked_Error += 1
            elif judge == ERROR:
                checkError +=1
                if exact == TRUE:
                    checkError_True += 1
                else:
                    checkError_Error += 1
            else: # 查的正确
                checkTrue += 1
                if exact == TRUE:
                    checkTrue_True += 1
                else:
                    checkTrue_Error += 1
            
            if exact == ERROR:
                exact_error += 1
            
            if judge == ERROR and exact == ERROR:
                good +=1
            if judge == ERROR and exact == TRUE:
                if verbose:
                    print(f"假阳性 {i} 行列 {c} 原值 {label_dirty_table.loc[i, c[1:]]} 正确值 {clean_table.loc[i, c[1:]]}")
                false_positive +=1
            if judge == TRUE and exact == ERROR:
                if verbose:
                    print(f"假阴性 {i} 行列 {c} 原值 {label_dirty_table.loc[i, c[1:]]} 正确值 {clean_table.loc[i, c[1:]]}")
                false_negative +=1

    print(f"共 {cell_num} 单元格，实际错误 {exact_error} 个")
    print(f"规则执行，认为 {checkTrue} 个单元格正确。实际上 {checkTrue_True} 个正确，{checkTrue_Error} 个错误")
    print(f"认为 {checkError} 个单元格错误。实际上 {checkError_True} 个正确，{checkError_Error} 个错误")
    print(f"有 {unchecked} 个单元格未被查验到，姑且认为正确的。实际上 {unchecked_True} 个正确，{unchecked_Error} 个错误")
    print(f"正确查出错误 {good}，假阳性 {false_positive}，假阴性 {false_negative}")
    
    recall = 0.0 if exact_error == 0 else good / exact_error
    precision = 0.0 if (good + false_positive) == 0 else good / (good + false_positive)
    f1 = 0.0 if (recall + precision) == 0 else 2 * recall * precision / (recall + precision)

    print(f"recall = {recall}")
    print(f"precision = {precision}")
    print(f"F1 = {f1}")

'''
返回 SQL、判断列名、X 涉及列名、规则类型

print(rule2sql("t(t0) ^ t0.a = '123' -> t0._b = '0'")) # [("SELECT aic FROM tab AS t0 WHERE t0.a = '123'", 'b', {'a'}, 1)]
print(rule2sql("t(t0) ^ t0.a = '123' -> t0._b = '1'")) # [("SELECT aic FROM tab AS t0 WHERE t0.a = '123'", 'b', {'a'}, 0)]

# [("SELECT t0.aic AS id0, t1.aic AS id1 FROM tab AS t0, tab AS t1 WHERE t0.a = t1.a AND t0.b = '36467' AND t1.b = '36467'", 'c', {'b', 'a'}, 2)]
print(rule2sql("t(t0) ^ t(t1) ^ t0.a = t1.a ^ t0.b = '36467' ^ t1.b = '36467' -> t0._c = t1._c"))

# [("SELECT aic FROM tab AS t0 WHERE t0.a = '123' AND t0.b = 'abc'", 'b', {'a'}, 3),
#  ("SELECT aic FROM tab AS t0 WHERE t0.a = '123' AND t0.b <> 'abc'", 'b', {'a'}, 4)]
print(rule2sql("t(t0) ^ t0.a = '123' -> t0.b = 'abc'"))

# [('SELECT t0.aic AS id0, t1.aic AS id1 FROM tab AS t0, tab AS t1 WHERE t0.a = t1.a AND t0.b <> t1.b', 'b', {'a'}, 5)]
print(rule2sql("t(t0) ^ t(t1) ^ t0.a = t1.a -> t0.b = t1.b"))

# [("SELECT t0.aic AS id0, t1.aic AS id1 FROM tab AS t0, tab AS t1 WHERE t0.a = t1.a AND t0.b = '12' AND t1.b = '12'", 'b', {'a', 'b'}, 6),
#  ("SELECT t0.aic AS id0, t1.aic AS id1 FROM tab AS t0, tab AS t1 WHERE t0.a = t1.a AND t0.b = '12' AND t1.b <> '12'", 'b', {'a', 'b'}, 7)]
print(rule2sql("t(t0) ^ t(t1) ^ t0.a = t1.a ^ t0.b = '12' -> t1.b = '12'"))
'''
def rule2sql(rule:str)->List[Tuple[str, str, Set[str], int]]:
    SQL = None
    WHERES,WHERES2 = [], None # 2 用来装第二个 SQL 如果有，此时 Y 为否定
    X_COLS = set()
    RULE_TYPE = None
    
    lr = rule.split(' -> ')
    left, right = lr[0], lr[1].strip()
    lhs = left.split(' ^ ')

    single_line = True
    for lh in lhs:
        lh = lh.strip()
        if ' ' in lh:
            # 存在空格，说明是谓词
            WHERES.append(lh)
            xCol = lh[lh.index('.')+1:lh.index(' ')]
            if xCol.startswith('_'):
                xCol = xCol[1:]
            X_COLS.add(xCol)
        else:
            # 不存在空格，是表声明
            if lh.endswith('(t1)'):
                single_line = False
    
    yCol = right[right.index('.')+1:right.index(' ')]
    if yCol.startswith('_'):
        # yCol 判别列 _xxx，不加入 WHERE
        yCol = yCol[1:]
        if right.endswith("'1'"):
            # 判正规则
            RULE_TYPE = RULE_JUST_TRUE
        elif right.endswith("'0'"):
            # 判错规则
            RULE_TYPE = RULE_JUST_ERROR
        else:
            # 多行规则，t0._xx = t1._xx，说明判断结果都要一样
            RULE_TYPE = RULE_ALL_SAME
    else:
        # yCol 是原始列 xxx
        if right.startswith('t1.'):
            # X -> t1.a=123
            RULE_TYPE = RULE_NORMAL_MULTI_JUST_T1_TRUE
            WHERES2 = WHERES[:]
            WHERES.append(right)
            eq = right.index('=')
            WHERES2.append(right[:eq] + '<>' + right[eq+1:])
        elif ' = t1.' in right:
            # X -> t0.a=t1.a
            RULE_TYPE = RULE_NORMAL_MULTI_OPPOSITE
            eq = right.index('=')
            WHERES.append(right[:eq] + '<>' + right[eq+1:])
        else:
            # X -> t0.a=123
            RULE_TYPE = RULE_NORMAL_SINGLE_JUST_TRUE
            WHERES2 = WHERES[:]
            WHERES.append(right)
            eq = right.index('=')
            WHERES2.append(right[:eq] + '<>' + right[eq+1:])
        

    if single_line:
        SQL = f"SELECT {AUTO_INCRE_COL} FROM {TEMP_TABLE_NAME} AS t0 WHERE "
    else:
        SQL = f"SELECT t0.{AUTO_INCRE_COL} AS id0, t1.{AUTO_INCRE_COL} AS id1 FROM {TEMP_TABLE_NAME} AS t0, {TEMP_TABLE_NAME} AS t1 WHERE "

    if RULE_TYPE == RULE_NORMAL_MULTI_JUST_T1_TRUE:
        return [
            (SQL + ' AND '.join(WHERES), yCol, X_COLS, RULE_TYPE),
            (SQL + ' AND '.join(WHERES2), yCol, X_COLS, RULE_NORMAL_MULTI_JUST_T1_ERROR)
        ]
    elif RULE_TYPE == RULE_NORMAL_SINGLE_JUST_TRUE:
        return [
            (SQL + ' AND '.join(WHERES), yCol, X_COLS, RULE_TYPE),
            (SQL + ' AND '.join(WHERES2), yCol, X_COLS, RULE_NORMAL_SINGLE_JUST_ERROR)
        ]
    else:
        return [(SQL + ' AND '.join(WHERES), yCol, X_COLS, RULE_TYPE)]

def rm(file:str):
    if os.path.exists(file):
        sleep(1e-10)
        os.remove(file)

# 返回是否设置值、是否交替
def _set_judge_value(rid, yCol, ruleId, rule, value, label_dirty_table:pd.DataFrame, validation_table:pd.DataFrame,
        start:int=0, end:int=None, override_to_true:bool=False, override_to_false:bool=False, record = sys.stdout)->Tuple[bool, bool]:
    if rid < start or (end is not None and rid >= end):
        return False, False
    is_true = (value == TRUE)
    des = "正确" if is_true else "错误"
    standard = validation_table.loc[rid, '_' + yCol]
    if standard != value:
        print(f'行 {rid} 判断错误，应为 {"正确" if standard == TRUE else "错误"} 判断为 {des}。{ruleId}.{rule}', file = record)

    old_value = label_dirty_table.loc[rid, '_' + yCol]
    if old_value == UNCHECK:
        print(f'首次认为 {yCol} 列 {rid} 是{des}。{ruleId}.{rule}', file = record)
        label_dirty_table.loc[rid, '_' + yCol] = value
        return True, False
    elif old_value == value:
        print(f'再次认为 {yCol} 列 {rid} 是{des}值。{ruleId}.{rule}', file = record)
        return False, False
    else: # 冲突
        if is_true:
            if override_to_true:
                print(f'之前认为 {yCol} 列 {rid} 是错误的，现改成正确。{ruleId}.{rule}', file = record)
                label_dirty_table.loc[rid, '_' + yCol] = TRUE
                return True, True
            else:
                print(f'之前认为 {yCol} 列 {rid} 是错误的，不与更改为正确。{ruleId}.{rule}', file = record)
                return False, False
        else:
            if override_to_false:
                print(f'之前认为 {yCol} 列 {rid} 是正确的，现改成错误。{ruleId}.{rule}', file = record)
                label_dirty_table.loc[rid, '_' + yCol] = ERROR
                return True, True
            else:
                print(f'之前认为 {yCol} 列 {rid} 是正确的，不与更改为错误。{ruleId}.{rule}', file = record)
                return False, False


'''规则执行
@param start 规则执行涉及开始行
@param end 规则执行涉及尾行，空值则不判断
@param override_to_true 是否允许将判断值更改为 TRUE
@param override_to_false 是否允许将判断值更改为 ERROR
@param strict mode: 进行判断时，所涉及其他单元格必须是 TRUE
'''
def rule_execute(label_dirty_table:pd.DataFrame, validation_table:pd.DataFrame, rules:List[str],
                 start:int=0, end:int=None, override_to_true:bool=False, override_to_false:bool=False, strict:bool=False):
    label_dirty_table = deepcopy(label_dirty_table)

    def strict_not_meet(rid, yCol, xCols, ruleId, rule):
        if strict:
            for xCol in xCols:
                if xCol != yCol:
                    value = label_dirty_table.loc[rid, '_' + xCol]
                    if value != TRUE:
                        # print(f'严格模式限制规则执行。{ruleId}.{rule}', file = recoed)
                        return False
        # print(f'通过严格模式规则执行。{ruleId}.{rule}', file = recoed)
        return True

    def rule_execute_sql(conn:sqlite3.Connection, recoed:io.TextIOWrapper):
        # 放入 DB
        label_dirty_table.to_sql(TEMP_TABLE_NAME, conn)
        # 查出正误的规则计数、没有查出正误的规则计数、正误交替计数
        modify, no_change, alternate = 0,0,0
        for i in tqdm(range(len(rules))):
            rule = rules[i].strip()
            SQLs = rule2sql(rule)
            trueCnt, errorCnt = 0, 0 # 查对、查错计数

            for SQL, yCol, xCols, type in SQLs:
                rows = conn.execute(SQL)
                if type == RULE_JUST_TRUE or type == RULE_NORMAL_SINGLE_JUST_TRUE:
                    for row in rows:
                        assert len(row) == 1
                        rid = row[0]
                        if not strict_not_meet(rid, yCol, xCols, i, rule):
                            continue
                        is_set, is_alternate = _set_judge_value(rid, yCol, i, rule, TRUE, label_dirty_table, validation_table, start, end, override_to_true, override_to_false, record)
                        trueCnt, alternate = trueCnt + (1 if is_set else 0), alternate + (1 if is_alternate else 0)
                elif type == RULE_JUST_ERROR or type == RULE_NORMAL_SINGLE_JUST_ERROR:
                    for row in rows:
                        assert len(row) == 1
                        rid = row[0]
                        if not strict_not_meet(rid, yCol, xCols, i, rule):
                            continue
                        is_set, is_alternate = _set_judge_value(rid, yCol, i, rule, ERROR, label_dirty_table, validation_table, start, end, override_to_true, override_to_false, record)
                        errorCnt, alternate = errorCnt + (1 if is_set else 0), alternate + (1 if is_alternate else 0)
                elif type == RULE_ALL_SAME:
                    rids = set() # 所有涉及的列
                    for row in rows:
                        assert len(row) == 2
                        for rid in row:
                            if not strict_not_meet(rid, yCol, xCols, i, rule):
                                continue
                            rids.add(rid)
                    # 找到众数，倾向于 TRUE 还是 ERROR。如果相同，倾向于 TRUE
                    true_num, error_num = 0, 0
                    for rid in rids:
                        old_value = label_dirty_table.loc[rid, '_' + yCol]
                        if old_value == TRUE:
                            true_num += 1
                        elif old_value == ERROR:
                            error_num += 1
                    target = TRUE if true_num >= error_num else ERROR
                    for rid in rids:
                        is_set, is_alternate = _set_judge_value(rid, yCol, i, rule, target, label_dirty_table, validation_table, start, end, override_to_true, override_to_false, record)
                        trueCnt, errorCnt, alternate = trueCnt + (1 if is_set and target == True else 0),errorCnt + (1 if is_set and target == ERROR else 0), alternate + (1 if is_alternate else 0)
                elif type == RULE_NORMAL_MULTI_OPPOSITE:
                    for row in rows:
                        assert len(row) == 2
                        left_rid, right_rid = row[0], row[1]
                        if not strict_not_meet(left_rid, yCol, xCols, i, rule):
                            continue
                        if not strict_not_meet(right_rid, yCol, xCols, i, rule):
                            continue
                        left_old_val, right_old_val = label_dirty_table.loc[left_rid, '_' + yCol], label_dirty_table.loc[right_rid, '_' + yCol]
                        if left_old_val == UNCHECK and right_old_val == UNCHECK:
                            continue
                        elif left_old_val != UNCHECK and right_old_val != UNCHECK:
                            continue
                        else:
                            if left_old_val != UNCHECK: # right_old_val 未设置
                                is_set, is_alternate = _set_judge_value(right_rid, yCol, i, rule, left_old_val, label_dirty_table, validation_table, start, end, override_to_true, override_to_false, record)
                                trueCnt, errorCnt, alternate = trueCnt + (1 if is_set and left_old_val == True else 0),errorCnt + (1 if is_set and left_old_val == ERROR else 0), alternate + (1 if is_alternate else 0)
                            else: # left_old_val 未设置
                                is_set, is_alternate = _set_judge_value(left_rid, yCol, i, rule, right_old_val, label_dirty_table, validation_table, start, end, override_to_true, override_to_false, record)
                                trueCnt, errorCnt, alternate = trueCnt + (1 if is_set and right_old_val == True else 0),errorCnt + (1 if is_set and right_old_val == ERROR else 0), alternate + (1 if is_alternate else 0)
                elif type == RULE_NORMAL_MULTI_JUST_T1_TRUE:
                    for row in rows:
                        assert len(row) == 2
                        if (not strict_not_meet(row[0], yCol, xCols, i, rule)) or (not strict_not_meet(row[1], yCol, xCols, i, rule)):
                            continue
                        rid = row[1]
                        is_set, is_alternate = _set_judge_value(rid, yCol, i, rule, TRUE, label_dirty_table, validation_table, start, end, override_to_true, override_to_false, record)
                        trueCnt, alternate = trueCnt + (1 if is_set else 0), alternate + (1 if is_alternate else 0)
                elif type == RULE_NORMAL_MULTI_JUST_T1_ERROR:
                    for row in rows:
                        assert len(row) == 2
                        if (not strict_not_meet(row[0], yCol, xCols, i, rule)) or (not strict_not_meet(row[1], yCol, xCols, i, rule)):
                            continue
                        rid = row[1]
                        is_set, is_alternate = _set_judge_value(rid, yCol, i, rule, ERROR, label_dirty_table, validation_table, start, end, override_to_true, override_to_false, record)
                        errorCnt, alternate = errorCnt + (1 if is_set else 0), alternate + (1 if is_alternate else 0)
                else:
                    raise Exception("Unknown type")

                        
            if trueCnt + errorCnt == 0:
                no_change +=1  
                print(f'没有查出任何正确和错误 {i}.{rule}', file = recoed)
            else:
                modify+=1
                print(f'查出正确 {trueCnt} 查出错误 {errorCnt} {i}.{rule}', file = recoed)

        print(f"一共执行了 {len(rules)} 条规则，其中 {no_change} 条没有查出任何正确和错误，{modify} 有查出正误")
        print(f"一共有 {alternate} 次正误替换。即前一个规则认为单元格是对/错的，后一个规则做出相反判断")

    rm(SQLITE3_TEMP_FILE)
    conn:sqlite3.Connection = sqlite3.connect(SQLITE3_TEMP_FILE)
    record:io.TextIOWrapper = open("record" + str(int(time())) + ".txt", mode = 'w', encoding='utf-8')
    try:
        rule_execute_sql(conn, record)
    finally:
        conn.close()
        rm(SQLITE3_TEMP_FILE)
        record.flush()
        record.close()
    
    return label_dirty_table

def rule_type(rule:str)->int:
    if ' ^ regular(' in rule:
        return RULE_REGEX
    else:
        # -1 表示普通类型，即单行、多行规则
        return -1

def regex_rule_execute(regex_rules:List[str], label_dirty_table:pd.DataFrame, validation_table:pd.DataFrame,
        start:int=0, end:int=None, override_to_true:bool=False, override_to_false:bool=False, strict=False)->pd.DataFrame:
    label_dirty_table = deepcopy(label_dirty_table)
    
    # 返回列名，正则表达式
    def extract_regex_str(regex_rule:str)->Tuple[str,str]:
        # tab(t0) ^ regular( t0.countyname , 're_str') -> true
        regular_info = regex_rule[regex_rule.index('^')+2:regex_rule.index('->')].strip()
        column = regular_info[regular_info.index('t0')+3:regular_info.index(',')].strip()
        regex_str = regular_info[regular_info.index('\'')+1:-2]
        print(f"正则规则 {regex_rule} 提取列名 {column} 正则 {regex_str}")
        return column, regex_str

    import re
    def execute(ruleId:int, regex_rule:str, regex_str:str, column:str):
        p = re.compile('^' + regex_str + '$')
        for rid in range(start, len(label_dirty_table) if end is None else end):
            cell_value:str = str(label_dirty_table.loc[rid, column])
            target = ERROR if p.match(cell_value) is None else TRUE
            _set_judge_value(rid, column, ruleId, regex_rule, target, label_dirty_table, validation_table, start, end, override_to_true, override_to_false)

    for i in range(len(regex_rules)):
        rule = regex_rules[i]
        column, regex_str = extract_regex_str(rule)
        execute(i, rule, regex_str, column)
    
    return label_dirty_table
    

if __name__ == '__main__':
    id_column = 'index'
    clean = load_csv("data/tax/clean.csv")
    dirty = load_csv("data/tax/dirty.csv")
    print(clean)
    print(dirty)

    dirty_label = append_judge_column(dirty, id_column)
    validation = label_by_clean(dirty_label, clean, id_column)
    print(validation)

    rees = [
        "tax(t0) ^ t0.child_exemp = '300' ^ t0.has_child = 'N' -> t0._has_child = '0'",
        "tax(t0) ^ t0.marital_status = 'M' ^ t0.single_exemp = '1500' -> t0._marital_status = '0'",
    ]

    regex_rules = [
        r"tax(t0) ^ regular( t0.l_name , '((?!'').)*') -> true",
        r"tax(t0) ^ regular( t0.f_name , '((?!'').)*') -> true",
        r"tax(t0) ^ regular( t0.state , '((?!-\*).)*') -> true",
        r"tax(t0) ^ regular( t0.city , '((?!-\*).)*') -> true",
        r"tax(t0) ^ regular( t0.single_exemp , '\d+') -> true",
        r"tax(t0) ^ regular( t0.child_exemp , '\d+') -> true",
    ]

    result = dirty_label
    result = regex_rule_execute(regex_rules, result, validation)
    result = rule_execute(result, validation, rees)
    check_result(result, validation, clean, id_column, verbose=True)
    # result = rule_execute(result, validation, all_same, start=600, strict=True)
    # check_result(result, validation, id_column, start=600)