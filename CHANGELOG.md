# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.10] - 2026-01-06
### Bug Fixes
- [`299bb15`](https://github.com/HydroRoll-Team/base-model/commit/299bb151d3db43f4b11f07b852aa9efbedde929a) - update version number to 0.1.10 in pyproject.toml *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*

### Chores
- [`d5fe97f`](https://github.com/HydroRoll-Team/base-model/commit/d5fe97fded336596d2991ffe6965669d14f7efb1) - update ONNX model files for improved performance *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*


## [v0.1.9] - 2026-01-05
### Bug Fixes
- [`9ae58c4`](https://github.com/HydroRoll-Team/base-model/commit/9ae58c4d2696c3c693330b9d7a5c1b738214d184) - update version number to 0.1.9 in pyproject.toml *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*

### Refactors
- [`077fb79`](https://github.com/HydroRoll-Team/base-model/commit/077fb791ca6734a7590d496c3e3fe5d25d3492e9) - clean up code formatting and improve print messages in main.py *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`911a561`](https://github.com/HydroRoll-Team/base-model/commit/911a5610703a4ee2d7438e1398c5bbe8079099cc) - remove unused fix_speaker function and clean up code in onnx_infer.py *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*


## [v0.1.8] - 2026-01-05
### New Features
- [`f51ebaf`](https://github.com/HydroRoll-Team/base-model/commit/f51ebaf36593dffb066ad3c4f7f98a0827d8f8e9) - improve code formatting and readability in conll_to_dataset.py *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`6713e38`](https://github.com/HydroRoll-Team/base-model/commit/6713e38407bdbc1495692c7e297c027a1dc3f612) - enhance code readability and formatting in onnx_infer.py *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*

### Bug Fixes
- [`1a6d6ed`](https://github.com/HydroRoll-Team/base-model/commit/1a6d6edbe14f00df021a1e826d8d30555f301e30) - simplify conditional assignment for entity text processing in test_onnx_only_infer.py *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`e4abb6c`](https://github.com/HydroRoll-Team/base-model/commit/e4abb6c2bc62dcb68d14fd4d44ff48d3773ed6d8) - update version number to 0.1.8 in __init__.py *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`182291c`](https://github.com/HydroRoll-Team/base-model/commit/182291cdbec8419da4097f8730249070f74bd04b) - update version number to 0.1.8 in pyproject.toml *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*

### Refactors
- [`65f48da`](https://github.com/HydroRoll-Team/base-model/commit/65f48da74e446df81b17d0cc9bf203b75947fff1) - improve code formatting and readability in utils/__init__.py *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`a9e98ae`](https://github.com/HydroRoll-Team/base-model/commit/a9e98ae197a49b8a6629601e3be7b9d0507eb6da) - streamline import statements and improve print messages in training module *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`db23ff8`](https://github.com/HydroRoll-Team/base-model/commit/db23ff87d996bfb9c63215b30f682df94f198299) - improve error handling and enhance code readability in inference module *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*


## [v0.1.7] - 2026-01-05
### New Features
- [`7b15d14`](https://github.com/HydroRoll-Team/base-model/commit/7b15d1470addfe6f7e7079c9f52c1ed7ded1484d) - add workflow to publish to TestPyPI on main branch push *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`f4f9c54`](https://github.com/HydroRoll-Team/base-model/commit/f4f9c541e9917fa614e6e1b8e737167f44c89c43) - add log processing and LLM annotation functionality *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`25380fb`](https://github.com/HydroRoll-Team/base-model/commit/25380fb4de77966a0f3d00681be25857c27b0869) - enhance log annotation process with improved concurrency and detailed prompt structure *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`df94eb6`](https://github.com/HydroRoll-Team/base-model/commit/df94eb6c125279a9c32bc85de8633371d50afbed) - update max_length parameter for TRPGParser and onnx_infer to improve text parsing capabilities *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`7531193`](https://github.com/HydroRoll-Team/base-model/commit/753119374a952cac55a0d87c13e5ae081e09de4b) - add max_length validation in TRPGParser to enforce input constraints *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`4c8ea1d`](https://github.com/HydroRoll-Team/base-model/commit/4c8ea1d68e823d3115c47a0fd5005490e5f2bb7a) - bump version to 0.1.7 for base-model-trpgner release *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*

### Bug Fixes
- [`14ba21c`](https://github.com/HydroRoll-Team/base-model/commit/14ba21c109916a9e2f119075b82deca8a81fa573) - remove unnecessary webui part in readme file *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`e84af26`](https://github.com/HydroRoll-Team/base-model/commit/e84af2657916f7a4a23d4dffadf58e52e1f4720d) - remove unnecessary permissions and update TestPyPI publish step *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`3c2f2d6`](https://github.com/HydroRoll-Team/base-model/commit/3c2f2d65e245fa6c2e7f8400e66012c47bdde15d) - remove unnecessary commands and comments in publish workflow *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`6fdf7ae`](https://github.com/HydroRoll-Team/base-model/commit/6fdf7ae765ee5e168f8426bd705a475d2843305d) - correct TestPyPI URL and update API token secret *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`8469222`](https://github.com/HydroRoll-Team/base-model/commit/8469222a80f204d9d0f6b5ba4dbdab4a0ff37649) - install twine before checking distribution in publish workflow *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`5b4c157`](https://github.com/HydroRoll-Team/base-model/commit/5b4c1572055c01c239b6bb648c70273167d6bffe) - add step to fix package metadata before publishing to TestPyPI *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`42fa9a4`](https://github.com/HydroRoll-Team/base-model/commit/42fa9a49d378087f0a5e9127a84dccfb2bd21016) - streamline package metadata modification process in publish workflow *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`9b4ec9c`](https://github.com/HydroRoll-Team/base-model/commit/9b4ec9c5fc256313e156c7d42b3104c711b15ecc) - update package name to use commit hash during TestPyPI publish *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`7871eb6`](https://github.com/HydroRoll-Team/base-model/commit/7871eb609d5505fba2d579ff29d9a30dcf01d35e) - add options to skip existing packages and enable verbose output during TestPyPI publish *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`219f4c3`](https://github.com/HydroRoll-Team/base-model/commit/219f4c3f4067f0b26699a7801775dc37f93429aa) - add options to skip existing packages and disable attestations during TestPyPI publish *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`6f9991d`](https://github.com/HydroRoll-Team/base-model/commit/6f9991d0200f04b1aedd198f67e6ac6537e21eb3) - update TestPyPI publish step to use environment variables and run command *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`2939917`](https://github.com/HydroRoll-Team/base-model/commit/29399171981b7bb1ba1f52db588a611521519212) - update TestPyPI publish step to use twine directly and ensure installation of twine *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`6d48ff2`](https://github.com/HydroRoll-Team/base-model/commit/6d48ff2af54c575224088c61cd61c14b9dee5001) - specify twine installation in publish workflow *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`c2b9ab2`](https://github.com/HydroRoll-Team/base-model/commit/c2b9ab21c93e148956a358635f1305029002b0ac) - add license-files entry in wheel build target *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`0189659`](https://github.com/HydroRoll-Team/base-model/commit/018965986690c16cd60d1f541901b6a8a85d46a5) - update license format in pyproject.toml *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`0f3a836`](https://github.com/HydroRoll-Team/base-model/commit/0f3a83652241d8800b75dc1db615edbb5e6bb36f) - replace COPYING file with LICENSE file in project structure *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`5dd32e5`](https://github.com/HydroRoll-Team/base-model/commit/5dd32e56b91bc5d0e5105a10e8f11bfafc8d0dee) - update TestPyPI upload command to use repository shorthand *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`cc7ac6e`](https://github.com/HydroRoll-Team/base-model/commit/cc7ac6e69b2dea53c66043604c14dc962a4a58a9) - update license format in pyproject.toml *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`3522b3e`](https://github.com/HydroRoll-Team/base-model/commit/3522b3efcc879a7238c6309ede2673f4bd75b27d) - update publish workflow to use uv for installing twine and publishing to TestPyPI *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`2164059`](https://github.com/HydroRoll-Team/base-model/commit/2164059395e962b1715fe0d3f46d8fc9f076eaab) - streamline TestPyPI publishing step in workflow *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`b4c0f9f`](https://github.com/HydroRoll-Team/base-model/commit/b4c0f9f844c0a44c0ef2b7b46a70f5aae3779d6e) - restructure TestPyPI publishing step in workflow *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`92daa23`](https://github.com/HydroRoll-Team/base-model/commit/92daa233aea36b28e457f0952237415ff1af66ae) - add uv sync command before publishing to TestPyPI *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`8f23487`](https://github.com/HydroRoll-Team/base-model/commit/8f23487d5c71b67428da163f01fc9c04f48ce274) - add debugging commands before publishing to TestPyPI *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`b1e11ce`](https://github.com/HydroRoll-Team/base-model/commit/b1e11ce7662586b880ce09fdd474934d11b4dbe2) - update debug command in TestPyPI publishing workflow *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`00518d0`](https://github.com/HydroRoll-Team/base-model/commit/00518d047b632274913909697969dc23cf880d6a) - update TestPyPI publishing step to use correct distribution path *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`4a7261d`](https://github.com/HydroRoll-Team/base-model/commit/4a7261d783b99467c01249e23163abf646ca30d4) - replace uv sync command with uv v in TestPyPI publishing step *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`54136b6`](https://github.com/HydroRoll-Team/base-model/commit/54136b6c4720ac1441f262b5d51c48d60b2d4e4e) - update TestPyPI publishing command to use relative path for distribution files *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*

### Chores
- [`d4480d0`](https://github.com/HydroRoll-Team/base-model/commit/d4480d00543ec63fc84b8142aa2728b24135c14c) - update README.md *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`910333f`](https://github.com/HydroRoll-Team/base-model/commit/910333fda7fc32ed426e96f11f01c76d6e95544b) - Update README.md *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*


## [v0.1.6] - 2025-12-30
### New Features
- [`d35712d`](https://github.com/HydroRoll-Team/base-model/commit/d35712d0f200b7862450b173a1bee95d1bd85dc8) - Update Python version requirement and add onnxscript dependency *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`e1444f4`](https://github.com/HydroRoll-Team/base-model/commit/e1444f4b283ebb628292c656c47ab0bf37739149) - Bump version to 0.1.6 in pyproject.toml *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*

### Refactors
- [`200ae0b`](https://github.com/HydroRoll-Team/base-model/commit/200ae0b8131367a55572cbfca00fbcf19257b84d) - Refactor code structure for improved readability and maintainability *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*


## [v0.1.5] - 2025-12-30
### New Features
- [`046eda6`](https://github.com/HydroRoll-Team/base-model/commit/046eda69af8ac163c4337eb63a544685716d97c3) - Add debug step for artifact structure and fix model zip path in release *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`2290d8b`](https://github.com/HydroRoll-Team/base-model/commit/2290d8bc56963e1ed50b6f2bc0e013d2b91ad34b) - Refine model packaging process to include only necessary inference files *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`1cda2f4`](https://github.com/HydroRoll-Team/base-model/commit/1cda2f4257f13a29224a421477b4d9f316927410) - Bump version to 0.1.5 in pyproject.toml *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*

### Bug Fixes
- [`e48db0f`](https://github.com/HydroRoll-Team/base-model/commit/e48db0fa0555886f23dd384e21cd16417dff4f7d) - Simplify artifact upload paths in GitHub release workflow *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*


## [v0.1.4] - 2025-12-30
### New Features
- [`a4dd04f`](https://github.com/HydroRoll-Team/base-model/commit/a4dd04f6e3af86ce3f96c7f7ebc88e195db366f4) - Enhance model download functionality to support zip file retrieval and extraction *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*
- [`6922fce`](https://github.com/HydroRoll-Team/base-model/commit/6922fcea0d567acfaf798c462f526fe43b72d351) - Bump version to 0.1.4 in pyproject.toml *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*

### Refactors
- [`18c946a`](https://github.com/HydroRoll-Team/base-model/commit/18c946aac2b0e16ec4e66bb4c40c62403af6f205) - Clean up publish workflow by removing Test PyPI steps and improving artifact packaging *(commit by [@HsiangNianian](https://github.com/HsiangNianian))*

[v0.1.4]: https://github.com/HydroRoll-Team/base-model/compare/v0.1.3...v0.1.4
[v0.1.5]: https://github.com/HydroRoll-Team/base-model/compare/v0.1.4...v0.1.5
[v0.1.6]: https://github.com/HydroRoll-Team/base-model/compare/v0.1.5...v0.1.6
[v0.1.7]: https://github.com/HydroRoll-Team/base-model/compare/v0.1.6...v0.1.7
[v0.1.8]: https://github.com/HydroRoll-Team/base-model/compare/v0.1.7...v0.1.8
[v0.1.9]: https://github.com/HydroRoll-Team/base-model/compare/v0.1.8...v0.1.9
[v0.1.10]: https://github.com/HydroRoll-Team/base-model/compare/v0.1.9...v0.1.10
