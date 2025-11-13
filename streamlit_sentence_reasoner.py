import streamlit as st
import pandas as pd
from sentence_reasoner_utils import (
    build_graph, parse_sentences_block, add_fact, graph_facts,
    apply_reasoning, apply_custom_rules, parse_rules, serialize_turtle, proof_to_text, EX
)

st.set_page_config(page_title='Sentence Reasoner (Ontology + Rules)', page_icon='🧩', layout='wide')
st.title('Sentence Reasoner — infer new facts from natural-language statements')
st.caption('Type simple sentences (controlled English), then infer new facts using either built-in or custom rules.')

if 'graph' not in st.session_state:
    st.session_state.graph = build_graph()
if 'derivations' not in st.session_state:
    st.session_state.derivations = {}
if 'custom_rules_text' not in st.session_state:
    st.session_state.custom_rules_text = '''
# Example custom rules (edit freely)
R1: IF parentOf(?x, ?y) THEN ancestorOf(?x, ?y)
R2: IF parentOf(?x, ?y) AND ancestorOf(?y, ?z) THEN ancestorOf(?x, ?z)
R3: IF manages(?x, ?y) THEN colleagueOf(?x, ?y)
R4: IF colleagueOf(?x, ?y) THEN colleagueOf(?y, ?x)
R5: IF worksAt(?x, ?o) AND worksAt(?y, ?o) AND neq(?x, ?y) THEN colleagueOf(?x, ?y)
R6: IF locatedIn(?x, ?y) AND locatedIn(?y, ?z) THEN locatedIn(?x, ?z)
'''

with st.expander('Examples', expanded=True):
    st.markdown('''
- Alice is the mother of Bob.
- Bob is the parent of Carol.
- Dave works at Contoso.
- Emma works at Contoso.
- Fiona manages Emma.
- Paris is in France.
- France is in Europe.
- ACME is located in Paris.
- Carol is a Engineer.
''')

st.subheader('Enter sentences')
text = st.text_area(
    'One sentence per line',
    height=200,
    placeholder='''Alice is the mother of Bob.
Bob is the parent of Carol.
Paris is in France.
France is in Europe.'''
)
colA, colB, colC = st.columns([1,1,1])
with colA:
    if st.button('Parse & Add Facts'):
        facts = parse_sentences_block(text)
        for (pred, subj, obj) in facts:
            add_fact(st.session_state.graph, pred, subj, obj)
        st.session_state.derivations = {}
        st.success(f'Added {len(facts)} facts from sentences.')
with colB:
    if st.button('Infer (Built-in Rules)'):
        deriv = apply_reasoning(st.session_state.graph)
        st.session_state.derivations = deriv
        st.success(f'Inference complete. Inferred {len(deriv)} new facts (built-in rules).')
with colC:
    if st.button('Clear Graph'):
        st.session_state.graph = build_graph()
        st.session_state.derivations = {}
        st.success('Cleared.')

st.divider()

# ----------------- Custom Rules Editor -----------------
st.subheader('Custom rules (editable)')
st.caption('Syntax: `R1: IF p(?x, ?y) AND q(?y, ?z) THEN r(?x, ?z)`; built-ins: `neq(?x, ?y)`')
st.session_state.custom_rules_text = st.text_area('Rules', value=st.session_state.custom_rules_text, height=200)
colR1, colR2, colR3 = st.columns([1,1,1])
with colR1:
    if st.button('Infer (Custom Rules)'):
        try:
            rules = parse_rules(st.session_state.custom_rules_text)
        except Exception as e:
            st.error(f'Rule parse error: {e}')
            rules = None
        if rules is not None:
            deriv = apply_custom_rules(st.session_state.graph, rules)
            st.session_state.derivations = deriv
            st.success(f'Inference complete. Inferred {len(deriv)} new facts (custom rules).')
with colR2:
    if st.button('Reset to Default Rules'):
        st.session_state.custom_rules_text = '''
R1: IF parentOf(?x, ?y) THEN ancestorOf(?x, ?y)
R2: IF parentOf(?x, ?y) AND ancestorOf(?y, ?z) THEN ancestorOf(?x, ?z)
R3: IF manages(?x, ?y) THEN colleagueOf(?x, ?y)
R4: IF colleagueOf(?x, ?y) THEN colleagueOf(?y, ?x)
R5: IF worksAt(?x, ?o) AND worksAt(?y, ?o) AND neq(?x, ?y) THEN colleagueOf(?x, ?y)
R6: IF locatedIn(?x, ?y) AND locatedIn(?y, ?z) THEN locatedIn(?x, ?z)
'''
        st.info('Restored default rules.')
with colR3:
    st.download_button('Download Rules (.txt)', data=st.session_state.custom_rules_text, file_name='custom_rules.txt', mime='text/plain', use_container_width=True)

# ----------------- Facts & Proofs -----------------
st.subheader('All facts (given + inferred)')
all_rows = graph_facts(st.session_state.graph)
df_all = pd.DataFrame(all_rows, columns=['predicate','subject','object'])
st.dataframe(df_all, use_container_width=True)

st.subheader('Inspect proof for an inferred fact')
if st.session_state.derivations:
    inferred_options = sorted(['{}({},{})'.format(p,s,o) for (p,s,o) in st.session_state.derivations.keys()])
    choice = st.selectbox('Choose inferred triple', options=inferred_options)
    if choice:
        p = choice.split('(')[0]
        inside = choice.split('(')[1].rstrip(')')
        s,o = inside.split(',')
        proof = proof_to_text((p,s,o), st.session_state.derivations)
        st.code(proof, language='text')
else:
    st.info('Run inference to see derivations/proofs.')

st.divider()
st.markdown('**Export graph**')
ttl_bytes = serialize_turtle(st.session_state.graph)
st.download_button('Download Turtle (.ttl)', data=ttl_bytes, file_name='sentence_reasoner_graph.ttl', mime='text/turtle', use_container_width=True)