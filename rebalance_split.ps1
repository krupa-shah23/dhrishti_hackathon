# Run this from the repo root: D:\College\Projects - Others\dhristi hackathon\dhrishti_hackathon
# Rebalances data\roboflow_export so valid/ and test/ actually contain phone + paper-chit boxes.
#
# Current state:
#   train : phone=136  paper-chit=111  (out of 474 images)
#   valid : phone=5    paper-chit=2    (out of 89 images)
#   test  : phone=0    paper-chit=0    (out of 30 images)   <-- broken
#
# Target after this script:
#   test  : +8 phone, +6 chit  ->  phone=8   paper-chit=6
#   valid : +10 phone, +8 chit ->  phone=15  paper-chit=10
#   train : loses 18 phone, 14 chit (still leaves 118 phone / 97 chit for training - plenty)

$root = ".\data\roboflow_export"
$imgExts = @(".jpg", ".jpeg", ".png")

function Get-ImagePath($base, $splitImagesDir) {
    foreach ($ext in $imgExts) {
        $p = Join-Path $splitImagesDir ($base + $ext)
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Move-Pair($base, $fromSplit, $toSplit) {
    $fromLabels = Join-Path $root "$fromSplit\labels"
    $fromImages = Join-Path $root "$fromSplit\images"
    $toLabels   = Join-Path $root "$toSplit\labels"
    $toImages   = Join-Path $root "$toSplit\images"

    $labelPath = Join-Path $fromLabels "$base.txt"
    $imagePath = Get-ImagePath $base $fromImages

    if (-not (Test-Path $labelPath)) { Write-Host "  MISSING label: $base"; return }
    if (-not $imagePath) { Write-Host "  MISSING image: $base"; return }

    Move-Item $labelPath (Join-Path $toLabels "$base.txt") -Force
    Move-Item $imagePath (Join-Path $toImages (Split-Path $imagePath -Leaf)) -Force
    Write-Host "  moved $base : $fromSplit -> $toSplit"
}

# Collect train label basenames by class, deduped
$trainLabels = Get-ChildItem (Join-Path $root "train\labels") -Filter *.txt
$phoneOnly = @()
$chitOnly = @()
foreach ($f in $trainLabels) {
    $content = Get-Content $f.FullName
    $hasPhone = $content -match "^1 "
    $hasChit  = $content -match "^0 "
    $base = [System.IO.Path]::GetFileNameWithoutExtension($f.Name)
    if ($hasPhone -and -not $hasChit) { $phoneOnly += $base }
    elseif ($hasChit -and -not $hasPhone) { $chitOnly += $base }
    # mixed-class images are skipped here to keep the moves simple/predictable
}

Write-Host "Available in train: phone-only=$($phoneOnly.Count)  chit-only=$($chitOnly.Count)"

$rand = New-Object System.Random
$phoneOnly = $phoneOnly | Sort-Object { $rand.Next() }
$chitOnly  = $chitOnly  | Sort-Object { $rand.Next() }

Write-Host "`nMoving to test (8 phone, 6 chit)..."
0..7  | ForEach-Object { Move-Pair $phoneOnly[$_] "train" "test" }
0..5  | ForEach-Object { Move-Pair $chitOnly[$_]  "train" "test" }

Write-Host "`nMoving to valid (10 phone, 8 chit)..."
8..17 | ForEach-Object { Move-Pair $phoneOnly[$_] "train" "valid" }
6..13 | ForEach-Object { Move-Pair $chitOnly[$_]  "train" "valid" }

Write-Host "`nDone. Re-run the split count check to confirm new totals."