# Autodesk PowerMill 2025

## Required build

- Product: Autodesk PowerMill Ultimate 2025
- Version: `25.0.0`

## Purchase and download

1. Purchase a PowerMill Ultimate subscription for the organization in [Autodesk Account](https://manage.autodesk.com/).
2. Assign the PowerMill entitlement to the Autodesk ID used for installation.
3. Sign in to Autodesk Account with that Autodesk ID.
4. Open **Products and Services > All Products and Services > PowerMill**.
5. Select version **2025**, operating system **Windows**, and the required language.
6. Select **Direct download** and download the PowerMill 2025 installation media. Direct download supplies the base `25.0.0` product without product updates.

Autodesk documents this account workflow in [Installation for individuals](https://www.autodesk.com/support/download-install/individuals/configure/install-your-product).

## Install and activate

1. Run the PowerMill 2025 installer as Administrator.
2. Start PowerMill and sign in with the assigned Autodesk ID.
3. Confirm that the assigned license opens the Ultimate edition.
4. Keep Autodesk credentials and license data outside this repository.

## Verify

Run the `verify` command in `meta.json`. It checks `C:\Program Files\Autodesk\PowerMill 2025\sys\exec64\pmill.exe` and the pinned ProductVersion.
