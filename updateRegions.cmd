    @echo off
    setlocal enabledelayedexpansion

    set regions="AECI", "AVA", "AVRN", "AZPS", "BANC", "BPAT", "CHPD", "CISO", "CPLE", "CPLW", "DEAA", "DOPD", "DUK", "EPE", "ERCO", "FMPP", "FPC", "FPL", "GCPD", "GRID", "GVL", "GWA", "HGMA", "HST", "IID", "IPCO", "ISNE", "JEA", "LDWP", "LGEE", "MISO", "NEVP", "NWMT", "NYIS", "PACE", "PACW", "PGE", "PJM", "PNM", "PSCO", "PSEI", "SC", "SCEG", "SCL", "SEC", "SEPA", "SOCO", "SPA", "SRP", "SWPP", "TAL", "TEC", "TEPC", "TIDC", "TPWR", "TVA", "WACM", "WALC", "WAUW", "WWA"

    for %%R in (%regions%) do (
        set region=%%~R
        set region=!region:"=!
        mkdir ./real_time/!region!
    )