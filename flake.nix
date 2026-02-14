{
  description = "NixOS WSL development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        p2n = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };

        # Блок исправлений для проблемных пакетов
        myOverrides = p2n.defaultPoetryOverrides.extend (self: super: {
          # Исправляем Pillow (ему нужны либы для работы с изображениями и pybind11)
          pillow = super.pillow.overridePythonAttrs (oldAttrs: {
            buildInputs = (oldAttrs.buildInputs or [ ]) ++ [ 
              pkgs.zlib 
              pkgs.libjpeg 
              pkgs.libpng 
              pkgs.libtiff 
              pkgs.freetype 
            ];
            nativeBuildInputs = (oldAttrs.nativeBuildInputs or [ ]) ++ [ 
              self.setuptools 
              self.pybind11 
              pkgs.pkg-config 
            ];
          });
        });

        myPythonEnv = p2n.mkPoetryEnv {
          projectDir = ./.;
          python = pkgs.python311;
          overrides = myOverrides;
          preferWheels = false; #Вроде это default значение
          extraPackages = (ps: [ ps.tkinter ]);
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ myPythonEnv pkgs.poetry pkgs.tcl pkgs.tk pkgs.xorg.libX11];

          shellHook = ''
            export DISPLAY=:0
            
            echo "🐍 Python: $(python --version)"
            echo "📦 Poetry: $(poetry --version)"
            echo "🎨 Tkinter status: $(python -c 'import tkinter; print("Ready")' 2>/dev/null || echo 'Not found')"
          '';
        };
      }
    );
}