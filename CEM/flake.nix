{
  inputs = {
    mach-nix.url = "github:DavHau/mach-nix?tag=3.5.0";
  };

  outputs = {self, nixpkgs, mach-nix }@inp:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems
        (system: f system (import nixpkgs {inherit system;}));
    in
    {
      defaultPackage = forAllSystems (system: pkgs: mach-nix.lib."${system}".mkPython {
        requirements = ''
          numpy ~= 1.23.3
        '';
        python = "python310";
      });
    };
}