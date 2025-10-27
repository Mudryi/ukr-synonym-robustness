import re


INTERCHANGEABLE_POS = {
    'PRCL': {'ADVB', 'CONJ'},
    'ADVB': {'PRCL', 'Prnt', 'PRED', 'GRND'},
    'Prnt': {'ADVB'},
    'CONJ': {'PRCL'},
    'ADJF': {'NPRO'},
    'NPRO': {'ADJF'},
    'PRED': {'ADVB'},
    'GRND': {'ADVB'}
    }

FORCE_ADVB = {'завжди', 'завше', 'навіщо', 'загалом'}

def get_pos_safe(parsing_result):
    if parsing_result.word in FORCE_ADVB:
        return 'ADVB'
    return parsing_result.tag.POS or str(parsing_result.tag).split(',')[0]


def get_correct_parsed_result(parsing_results, target_pos, target_gender=None):
    acceptable_pos = {target_pos} | INTERCHANGEABLE_POS.get(target_pos, set())

    # Step 1: Exact POS + gender match
    for result in parsing_results:
        if get_pos_safe(result) in acceptable_pos and target_gender and result.tag.gender == target_gender:
            return result

    # Step 2: POS match without gender
    for result in parsing_results:
        if get_pos_safe(result) in acceptable_pos:
            return result
    
    # Step 3: Special fallback: allow ADJF neut-nomn as ADVB
    if target_pos == 'ADVB':
        for result in parsing_results:
            if result.tag.POS == 'ADJF' and {'neut', 'nomn'}.issubset(result.tag.grammemes):
                return result
    
    if target_pos == 'ADJF':
        for result in parsing_results:
            if result.tag.POS == 'ADVB':
                return result
            
    # Step 4: Special FORCE_ADVB override
    for result in parsing_results:
        if result.word.lower() in FORCE_ADVB and target_pos == 'ADVB':
            return result

    return None


def tokenize_ukrainian(text):
    pattern = r"(\s+|[^\w\s']+|[\w']+)"
    tokens = re.findall(pattern, text)
    
    final_tokens = []
    i = 0
    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and tokens[i].isalpha()
            and tokens[i+1] == "'"
            and tokens[i+2].isalpha()
        ):
            final_tokens.append(tokens[i] + tokens[i+1] + tokens[i+2])
            i += 3
        else:
            final_tokens.append(tokens[i])
            i += 1
    return final_tokens


def lower_grammar_restrictions(grammemes):
    grammemes_to_remove = ['Refl', 'compb', 'COMP', 'Qual']
    
    new_grammemes = set(item for item in list(grammemes) if item not in grammemes_to_remove)
    return new_grammemes


def stepwise_inflect(parse, target_grammemes, 
                     preferred_order=('plur', 'sing', 'femn', 'masc', 'neut', 'nomn', 'accs', 'gent', 'datv', 'loct', 'ablt', 'anim', 'inan')):
    current = parse
    applied = set()

    sorted_grammemes = sorted(target_grammemes, key=lambda g: preferred_order.index(g) if g in preferred_order else len(preferred_order))
    
    for gram in sorted_grammemes:
        attempt = current.inflect(applied | {gram})
        if attempt is not None:
            current = attempt
            applied |= {gram}
    return current


def replace_word(sentence, target, replacement, morph, debug=False):
    target_normal_forms = [parsing_result.normal_form for parsing_result in morph.parse(target)]
    
    tokens = tokenize_ukrainian(sentence)
    
    replaced = False
    new_tokens = []

    for i, token in enumerate(tokens):        
        if re.match(r'\w+', token):
            parse_options = morph.parse(token)
            replaced_curr_token = False
            
            for parsed_word in parse_options:
                if parsed_word.normal_form in target_normal_forms:
                    target_pos = get_pos_safe(parsed_word)
                    target_gender = parsed_word.tag.gender if target_pos in ('NOUN', 'ADJF') else None

                    replacement_parsed = morph.parse(replacement)
                    matched_replacement = get_correct_parsed_result(replacement_parsed, target_pos, target_gender)

                    if not matched_replacement:
                        if debug:
                            print(f"bad match {target} -> {replacement}")
                            print(target_pos, target_gender)
                            print(replacement_parsed)
                        continue
                    
                    grammemes = parsed_word.tag.grammemes 
                    cleaned_grammemes = lower_grammar_restrictions(grammemes)
                    replacement_inflected = stepwise_inflect(matched_replacement, cleaned_grammemes)
                    
                    if not replacement_inflected:
                        if debug:
                            print(f"bad inflect {target} -> {replacement}")
                        continue
                    
                    replacement_word = replacement_inflected.word
                    
                    if token.istitle():
                        replacement_word = replacement_word.capitalize()
                    
                    new_tokens.append(replacement_word)
                    replaced = True
                    replaced_curr_token = True
                    break
            
            if not replaced_curr_token:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    
    if not replaced:
        if debug:
            print(f'no replacement for {target}')
        return None
    return new_tokens