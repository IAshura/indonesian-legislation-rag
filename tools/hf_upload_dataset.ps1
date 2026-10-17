<#
.SYNOPSIS
  Upload Indonesian Legislation RAG artifacts to Hugging Face (selective, safe).
.USAGE
  1. Install CLI once:   powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
  2. Login once:         hf auth login        (token with WRITE access, from hf.co/settings/tokens)
  3. Run from repo root: powershell -File tools\hf_upload_dataset.ps1
                          add -IncludeCorpus to also push the raw corpus (units.csv + texts)
#>
param(
  [string]$Repo = "Fayizz/indonesian-legislation-norms",
  [switch]$IncludeCorpus
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot   # repo root (this script sits in tools/)

function Push-File {
  param([string]$Local, [string]$Remote)
  $full = Join-Path $Root $Local
  if (-not (Test-Path -LiteralPath $full)) {
    Write-Host "  [SKIP] not found: $Local" -ForegroundColor Yellow
    return
  }
  Write-Host "  [UP]   $Local -> $Remote" -ForegroundColor Cyan
  hf upload $Repo $full $Remote --repo-type dataset --quiet
  if ($LASTEXITCODE -ne 0) { throw "upload failed: $Local" }
}

Write-Host "== 1/4 Dataset card ==========================="
Push-File "HF_DATASET_CARD.md" "README.md"

Write-Host "== 2/4 Build artifacts (jsonl + manifest) ====="
Push-File "pipeline\result\chunks.jsonl"          "chunks.jsonl"
Push-File "pipeline\result\parents.jsonl"         "parents.jsonl"
Push-File "pipeline\result\parents_windows.jsonl" "parents_windows.jsonl"
Push-File "pipeline\result\documents.jsonl"       "documents.jsonl"
Push-File "pipeline\result\validity.jsonl"        "validity.jsonl"
Push-File "pipeline\result\quarantine.jsonl"      "quarantine.jsonl"
Push-File "pipeline\result\manifest.json"         "manifest.json"
Push-File "pipeline\result\lexical\norma_tokens.txt" "lexical/norma_tokens.txt"

Write-Host "== 3/4 Embeddings ============================="
foreach ($model in @("qwen3-06b", "qwen3-4b")) {
  Push-File "pipeline\result\embeddings\$model\embeddings.npy" "embeddings/$model/embeddings.npy"
  Push-File "pipeline\result\embeddings\$model\ids.jsonl"      "embeddings/$model/ids.jsonl"
  Push-File "pipeline\result\embeddings\$model\meta.json"      "embeddings/$model/meta.json"
}

Write-Host "== 4/4 Raw corpus (optional) =================="
if ($IncludeCorpus) {
  Push-File "data\corpus\units.csv"     "corpus/units.csv"
  Push-File "data\corpus\documents.csv" "corpus/documents.csv"
  Push-File "data\corpus\checksums.md5" "corpus/checksums.md5"
  Push-File "data\corpus\README-corpus.md" "corpus/README.md"
  Get-ChildItem -LiteralPath (Join-Path $Root "data\corpus\texts") -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring((Join-Path $Root "data\corpus\texts").Length + 1).Replace("\", "/")
    Push-File "data\corpus\texts\$rel" "corpus/texts/$rel"
  }
} else {
  Write-Host "  (dilewati - jalankan dengan -IncludeCorpus bila mau mengunggah korpus mentah)"
}

Write-Host ""
Write-Host "Selesai. Cek di: https://huggingface.co/datasets/$Repo" -ForegroundColor Green
