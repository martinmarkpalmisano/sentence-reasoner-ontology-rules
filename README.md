
# Sentence Reasoner — Ontology + Rules from Natural Language

This sample converts simple English sentences into RDF facts and applies forward-chaining rules to infer new facts (e.g., transitive *ancestorOf*, symmetric *colleagueOf*, transitive *locatedIn*).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_sentence_reasoner.txt
streamlit run streamlit_sentence_reasoner.py
```

## Supported sentence patterns
- `<A> is the parent/mother/father of <B>.` → `parentOf(A,B)`
- `<A> works at <B>.` → `worksAt(A,B)`
- `<A> manages <B>.` → `manages(A,B)`
- `<A> is a colleague of <B>.` → `colleagueOf(A,B)`
- `<A> is located in <B>.` / `<A> is in <B>.` / `<A> is part of <B>.` → `locatedIn(A,B)`
- `<A> is a <Class>.` → `rdf:type(A, Class)` (use `isa(A,Class)` in rules)

## Rules
Two options:
1. **Built-in rules** (R1–R6: ancestor/colleague/location transitivity & symmetry)
2. **Custom rules editor** in the UI. Syntax:
   ```
   R1: IF parentOf(?x, ?y) THEN ancestorOf(?x, ?y)
   R2: IF parentOf(?x, ?y) AND ancestorOf(?y, ?z) THEN ancestorOf(?x, ?z)
   R3: IF manages(?x, ?y) THEN colleagueOf(?x, ?y)
   R4: IF colleagueOf(?x, ?y) THEN colleagueOf(?y, ?x)
   R5: IF worksAt(?x, ?o) AND worksAt(?y, ?o) AND neq(?x, ?y) THEN colleagueOf(?x, ?y)
   R6: IF locatedIn(?x, ?y) AND locatedIn(?y, ?z) THEN locatedIn(?x, ?z)
   ```
   - Variables start with `?` (e.g., `?x`).
   - **Built-in guards**: `neq(?x, ?y)` enforces `?x != ?y`.
   - Predicates are binary. Use `isa(A,Class)` to reference RDF type.

## Proofs
After running inference, select any inferred triple to see a proof tree showing the rule and source facts used to derive it.

## Notes
- The rule engine is illustrative and operates in-memory. For large graphs, consider rule-capable stores (Jena/RDFox/Stardog) or SPARQL property paths.
