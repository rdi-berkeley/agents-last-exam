# Steinberg Cubase 15

## Required build

- Product: Steinberg Cubase Pro 15
- Version: `15.0.20`
- Installer: `Cubase_15.0.20_Installer_win64.zip`

## Purchase and download

1. Purchase Cubase Pro 15 from the [Steinberg online shop](https://www.steinberg.net/cubase/).
2. Redeem the supplied Download Access Code into the Steinberg ID used for installation.
3. Install the [Steinberg Download Assistant for Windows](https://www.steinberg.net/sda-win).
4. Sign in to Steinberg Download Assistant with that Steinberg ID.
5. Open **Cubase > Cubase Pro 15** and download the **Cubase 15 Application** installer.
6. Retain `Cubase_15.0.20_Installer_win64.zip` securely for future reinstalls.

Steinberg documents Download Assistant as the account-based source for full installers on its [Download Assistant page](https://o.steinberg.net/en/support/downloads/steinberg_download_assistant.html). The Cubase 15 installer is shared across editions; the activated license selects the Pro edition.

## Install and activate

1. Extract `Cubase_15.0.20_Installer_win64.zip`.
2. Run the Windows installer as Administrator.
3. Open Steinberg Activation Manager and sign in with the licensed Steinberg ID.
4. Activate the purchased Cubase Pro 15 license.
5. Disable automatic application updates to preserve the pinned build.
6. Keep Steinberg credentials and activation codes outside this repository.

## Verify

Run the `verify` command in `meta.json`. It checks `C:\Program Files\Steinberg\Cubase 15\Cubase15.exe` and the pinned ProductVersion.
