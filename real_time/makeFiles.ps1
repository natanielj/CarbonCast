$US_REGIONS = @(
    "AECI", "AVA", "AVRN", "AZPS", "BANC", "BPAT", "CHPD", "CISO", "CPLE", "CPLW",
    "DEAA", "DOPD", "DUK", "EPE", "ERCO", "FMPP", "FPC", "FPL", "GCPD", "GRID",
    "GVL", "GWA", "HGMA", "HST", "IID", "IPCO", "ISNE", "JEA", "LDWP", "LGEE",
    "MISO", "NEVP", "NWMT", "NYIS", "PACE", "PACW", "PGE", "PJM", "PNM", "PSCO",
    "PSEI", "SC", "SCEG", "SCL", "SEC", "SEPA", "SOCO", "SPA", "SRP", "SWPP",
    "TAL", "TEC", "TEPC", "TIDC", "TPWR", "TVA", "WACM", "WALC", "WAUW", "WWA"
)

foreach ($region in $US_REGIONS) {
    # If the folder doesn't already exist, create it
    if (-not (Test-Path -Path $region -PathType Container)) {
        New-Item -Path $region -ItemType Directory | Out-Null
        Write-Host "Created folder: $region"
    }
    else {
        Write-Host "Folder already exists: $region"
    }
}