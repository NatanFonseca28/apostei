---
description: Commit and Push to GitHub Repository
---

Este workflow inicializa o repositório Git, faz o commit inicial com o README.md e envia para o GitHub.

// turbo-all

1. Cria o arquivo README.md (ou anexa o título se já existir)
```powershell
echo "# apostei" >> README.md
```

2. Inicializa o repositório Git local
```powershell
git init
```

3. Adiciona o README.md à staging area
```powershell
git add README.md
```

**Opcional:** *Se você quiser adicionar todos os arquivos do projeto (não apenas o README), você pode rodar esse comando extra no terminal: `git add .`*

4. Cria o primeiro commit
```powershell
git commit -m "first commit"
```

5. Renomeia a branch principal para 'main'
```powershell
git branch -M main
```

6. Adiciona o repositório remoto apontando para o seu GitHub
```powershell
git remote add origin https://github.com/NatanFonseca28/apostei.git
```

7. Envia o código para o repositório remoto (push)
```powershell
git push -u origin main
```
