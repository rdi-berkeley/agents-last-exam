# Autodesk Civil 3D 2024

## Required build

- Product: Autodesk Civil 3D 2024
- Update: Civil 3D 2024.4.6
- Registry version: `13.6.2123.0`

## Purchase and download

1. Purchase a Civil 3D subscription for the organization in [Autodesk Account](https://manage.autodesk.com/).
2. Assign the Civil 3D entitlement to the Autodesk ID used for installation.
3. Sign in to Autodesk Account with that Autodesk ID.
4. Open **Products and Services > All Products and Services > Civil 3D**.
5. Select version **2024**, operating system **Windows**, and the required language.
6. Select **Direct download** and download the Civil 3D 2024 installation media. Direct download supplies the base product without product updates.
7. Open **Products and Services > Product Updates**.
8. Search for **Civil 3D 2024.4.6 Update** and select **Download**.

Autodesk documents the account download methods in [Installation for individuals](https://www.autodesk.com/support/download-install/individuals/configure/install-your-product) and the update workflow in [How to download product updates from Autodesk Account](https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/How-to-Access-Product-Updates-in-Autodesk-Accounts.html).

## Install and activate

1. Run the Civil 3D 2024 installer as Administrator.
2. Run the Civil 3D 2024.4.6 update as Administrator.
3. Start Civil 3D and sign in with the assigned Autodesk ID.
4. Keep Autodesk credentials and license data outside this repository.

## Verify

Run the `verify` command in `meta.json`. It checks `C:\Program Files\Autodesk\AutoCAD 2024\acad.exe` and the pinned ProductVersion.
