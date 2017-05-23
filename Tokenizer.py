import re
import html
import string

EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )"""
    ,
    # Emoticons:
    EMOTICONS
    ,
    # HTML tags:
    r"""<[^>]+>"""
    ,
    # Twitter username:
    r"""(?:@[\w_]+)"""
    ,
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
)

word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)
emoticons_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)

html_special = {'strong', 'b', 'i', 'em'}
special_re = re.compile(r'(?!</?(' + '|'.join(html_special) + ')+?>)(<[^<]+?>)', re.VERBOSE | re.I | re.UNICODE)

punctuation = {',', '.', ';', '...', '?', '!', ":"}
# Negation words given by
negation_words = {'never', 'no', 'nothing', 'nowhere', 'noone', 'none', 'not', 'havent', 'hasnt', 'hadnt', 'cant',
                  'couldnt', 'shouldnt', 'wont', 'wouldnt', 'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'aint', 'n\'t', 
                  "haven't", "hasn't", "hadn't", "can't", "couldn't", "shouldn't", "won't", "wouldn't", "don't", "didn't", 
                  "isn't", "aren't", "ain't", "neither", "nor"}

negate_re = re.compile(r'(' + '|'.join(negation_words) + ')', re.VERBOSE | re.I | re.UNICODE)

special_start_re = re.compile(r'<(' + '|'.join(html_special) + ')+?>', re.VERBOSE | re.I | re.UNICODE)
special_end_re = re.compile(r'</(' + '|'.join(html_special) + ')+?>', re.VERBOSE | re.I | re.UNICODE)

punctuation = string.punctuation

from nltk.corpus import stopwords
stop = stopwords.words('english') + list(punctuation)

class Tokenizer(object):
    def __init__(self, remove_html=True, negate=True, html_special=True):
        self.remove_html = remove_html
        self.negate = negate
        self.html_special = html_special

    def tokenize(self, string):
        # Replace unicode characters with html code
        string = html.unescape(string)

        # Tokenization - based on Potts' sentiment-aware tokenizer
        words = word_re.findall(string)
        words = map((lambda x: x if emoticons_re.search(x) else x.lower()), words)

        # Removing html tags (except tags given in html_special list)
        if self.remove_html:
            words = [x for x in words if not special_re.match(x)]

        if self.html_special:
            # Convert to uppercase all tokens between special html
            counter = 0
            i = 0
            while i < len(words):
                word = words[i]
                if special_start_re.match(word):
                    counter += 1
                    words.pop(i)
                    continue
                elif special_end_re.match(word):
                    counter -= 1
                    words.pop(i)
                    continue
                elif counter > 0:
                    words[i] = word.upper()
                i += 1

        if self.negate:
            # Negate words between negation and punctuation
            negate_mod = False
            for i in range(len(words)):
                word = words[i]
                if negate_re.match(word):
                    negate_mod = True
                    continue
                elif word in punctuation:
                    negate_mod = False
                    continue
                if negate_mod:
                    words[i] = word + "_NEG"

        return list(words)
