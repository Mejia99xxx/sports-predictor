content = open('backend/templates/index.html', encoding='utf-8').read()

checks = [
    ('method="POST"' in content, 'form has method POST'),
    ('action="/predecir"' in content, 'form has action /predecir'),
    ('name="modo" value="manual"' in content, 'hidden modo field'),
    ('name="posesion_local"' in content, 'posesion_local field'),
    ('name="posesion_visitante"' in content, 'posesion_visitante field'),
    ('name="tiros_local"' in content, 'tiros_local field'),
    ('name="tiros_visitante"' in content, 'tiros_visitante field'),
    ('name="faltas_local"' in content, 'faltas_local field'),
    ('name="faltas_visitante"' in content, 'faltas_visitante field'),
    ('name="tarjetas_local"' in content, 'tarjetas_local field'),
    ('name="tarjetas_visitante"' in content, 'tarjetas_visitante field'),
    ('type="submit"' in content, 'submit button present'),
    ('tiros_arco' not in content, 'no tiros_arco fields (excluded correctly)'),
    ('script.js' not in content, 'no script.js dependency (submit is native HTML)'),
]

all_pass = True
for result, desc in checks:
    status = 'PASS' if result else 'FAIL'
    if not result:
        all_pass = False
    print(f'{status}: {desc}')

print()
print('All checks passed!' if all_pass else 'SOME CHECKS FAILED!')
