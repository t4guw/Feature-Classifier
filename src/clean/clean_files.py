import re


# parentheses = re.compile(r'(\(.*?\))|(\[.*?\])|(\{.*?\})')
function_re = re.compile(r'[^!@#$%^&*()[\]{}|\\"\'+=;:<>?/`~\s]+?\(.*?\)') # functions like "findMin()" or "truncate(string input)"
brackets_re = re.compile(r' \[.*?\]') # standalone brackets like "[1, 2, 3]", but not array-like calls such as "arr[i]"
curlybraces_re = re.compile(r' \{.*?\}') # standalone curly braces like "{1, 2, 3}"
indexcalls_re = re.compile(r'[^ ]+?\[.*?\]') # array-like indexing calls such as "arr[i]"
doublequotes_re = re.compile(r' \".*?\"') # standalone quotations like "text in quotes"
singlequotes_re = re.compile(r" \'.*?\'") # standalone single quotations like 'text in single-quotes'


statement = [
'Design a data structure that supports adding new words and finding if a string matches any previously added string.', \
'Implement the WordDictionary class:', \
'	WordDictionary() Initializes the object.', \
'	void addWord(word) Adds word to the data structure, it can be matched later.', \
"	bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.", \
'Most be O(nlog(n)).', \
'Do not use C++ STL for addWord(word).']

statement2 = ['WordDictionary()']


# These functions are defined from top to bottom in the order they should be called in.
# for instance, call remove_indents_and_newlines() first

def remove_indents_and_newlines(statement_lines):
    return [line.strip(' ').strip('\t').strip('\n') for line in statement_lines]


def remove_blank_lines(statement_lines):
    new_lines = []
    for line in statement_lines:
        if line.replace(' ', '') != '':
            new_lines.append(line)
    return new_lines


def remove_lines_shorter_than(length, statement_lines):
    new_lines = []
    for line in statement_lines:
        if len(line) >= length:
            new_lines.append(line)
    return new_lines


def remove_lines_with_too_many_consecutive_spaces(spaces, statement_lines):
    new_lines = []
    for line in statement_lines:
        if (' ' * spaces) not in line:
            new_lines.append(line)
    return new_lines


def enumerate_functions(statement_lines):
    func_map = {}
    new_lines = []
    for line in statement_lines:
        for tok in line.split():
            curr = function_re.search(tok)

            if curr != None and curr.group()[0] != 'O' and curr.group() not in func_map:
                func_map[curr.group()] = 'function ' + str(len(func_map.keys()))
        
        for func_str in func_map.keys():
            line = line.replace(func_str, func_map[func_str])
        
        new_lines.append(line)

    return new_lines


def compile_statements(file_lines):
    statements_list = []
    curr_statement = ['']
    problem_num = -1
    get_num = False

    for line in file_lines:
        if line.replace('\n', '') == '~~~STATEMENT_START~~~' and len(curr_statement) > 0:
            statements_list.append((problem_num, curr_statement))
            curr_statement = []
            get_num = True
            continue
        if get_num:
            problem_num = int(line.replace('\n', ''))
            get_num = False
            continue
        curr_statement.append(line)

    if statements_list[0][0] == -1:
        del statements_list[0]
    
    statements_list.append((problem_num, curr_statement))
    return statements_list


def replace_square_brackets_with(replacement_word, statement_lines):
    return [re.sub(brackets_re, ' ' + replacement_word, line) for line in statement_lines]


def replace_curly_braces_with(replacement_word, statement_lines):
    return [re.sub(curlybraces_re, ' ' + replacement_word, line) for line in statement_lines]


def replace_single_quotes_with(replacement_word, statement_lines):
    return [re.sub(singlequotes_re, ' ' + replacement_word, line) for line in statement_lines]


def replace_double_quotes_with(replacement_word, statement_lines):
    return [re.sub(doublequotes_re, ' ' + replacement_word, line) for line in statement_lines]


def replace_index_calls_with(replacement_word, statement_lines):
    return [re.sub(indexcalls_re, replacement_word, line) for line in statement_lines]


statements_file = open('statements.txt', 'r')
clean_statements_file = open('statements_cleaned1.txt', 'w')

statements_list = compile_statements(statements_file.readlines())

for statement in statements_list:
    problem_num = statement[0]
    statement_lines = statement[1]

    statement_lines = remove_indents_and_newlines(statement_lines)
    statement_lines = remove_blank_lines(statement_lines)
    statement_lines = remove_lines_shorter_than(25, statement_lines)
    statement_lines = remove_lines_with_too_many_consecutive_spaces(3, statement_lines)
    statement_lines = enumerate_functions(statement_lines)

    statement_lines = replace_square_brackets_with('array', statement_lines)
    statement_lines = replace_curly_braces_with('set', statement_lines)
    statement_lines = replace_single_quotes_with('value', statement_lines)
    statement_lines = replace_double_quotes_with('value', statement_lines)
    statement_lines = replace_index_calls_with('array of index i', statement_lines)

    clean_statements_file.write('~~~STATEMENT_START~~~\n')
    clean_statements_file.write(str(problem_num) + '\n')

    for line in statement_lines:
        clean_statements_file.write(line + '\n')
