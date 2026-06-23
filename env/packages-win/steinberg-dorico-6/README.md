# Steinberg Dorico 6

## Required build

- Product: Steinberg Dorico Pro 6
- Version: `6.2.10`
- Installer: `Dorico_6.2.10_Installer_win64.zip`

## Purchase and download

1. Purchase Dorico Pro 6 from the [Steinberg online shop](https://www.steinberg.net/dorico/).
2. Redeem the supplied Download Access Code into the Steinberg ID used for installation.
3. Download the pinned Windows installer from Steinberg: [Dorico_6.2.10_Installer_win64.zip](https://download.steinberg.net/automated_updates/sda_downloads/aa79cebc-bdea-4359-98f9-c27535dc1b16/Dorico_6.2.10_Installer_win64.zip).
4. Install the [Steinberg Download Assistant for Windows](https://www.steinberg.net/sda-win) to obtain Steinberg Activation Manager, Steinberg Library Manager, Steinberg MediaBay, HALion Sonic, and the licensed Dorico sound content.

The [official Dorico 6 downloads page](https://o.steinberg.net/en/support/downloads/dorico_6.html) lists the `6.2.10` Windows installer in its older installers table.

## Install and activate

1. Extract `Dorico_6.2.10_Installer_win64.zip`.
2. Run the Dorico installer as Administrator.
3. Install HALion Sonic and the required Dorico sound content through Steinberg Download Assistant.
4. Open Steinberg Activation Manager and sign in with the licensed Steinberg ID.
5. Activate the purchased Dorico Pro 6 license.
6. Disable automatic application updates to preserve the pinned build.
7. Keep Steinberg credentials and activation codes outside this repository.

## Verify

Run the `verify` command in `meta.json`. It checks `C:\Program Files\Steinberg\Dorico6\Dorico6.exe` and the pinned ProductVersion.
