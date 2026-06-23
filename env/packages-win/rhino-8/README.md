# Rhino 8

## Required build

- Product: Rhino 8 for Windows
- Version: `8.26.25349.19001`

## Purchase and download

1. Purchase a Rhino 8 license from [McNeel](https://www.rhino3d.com/sales/).
2. Assign the license to the Rhino account used for installation or add it to the organization's Cloud Zoo team.
3. Download the pinned Windows installer from McNeel: [rhino_en-us_8.26.25349.19001.exe](https://files.mcneel.com/dujour/exe/20251215/rhino_en-us_8.26.25349.19001.exe).

## Install and activate

1. Run `rhino_en-us_8.26.25349.19001.exe` as Administrator.
2. Start Rhino 8.
3. Sign in to the assigned Rhino account and select the purchased local, Cloud Zoo, or LAN Zoo license.
4. Disable automatic application updates to preserve the pinned build.
5. Keep license keys and Rhino account credentials outside this repository.

## Verify

Run the `verify` command in `meta.json`. It checks `C:\Program Files\Rhino 8\System\Rhino.exe` and the pinned ProductVersion.
