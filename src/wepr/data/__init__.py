"""Data acquisition, loading and validation.

Import order of responsibility: `fetch` talks to the network and writes raw files,
`load` turns raw files into tidy frames, `validate` checks the result. Nothing here
runs on import.
"""
