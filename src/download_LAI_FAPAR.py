import os
import openeo
from src.utils import ensure_path_exists

class ProductDownloader:
    def __init__(
        self, 
        products: dict, 
        tiles: dict, 
        dates: list[str], 
        save_path: str, 
        backend_url: str = "https://openeo.vito.be"
    ):
        """
        Initializes the downloader with configuration for products, tile regions, and dates.
        """
        self.products = products
        self.tiles = tiles
        self.dates = dates
        self.save_path = save_path
        self.backend_url = backend_url

        ensure_path_exists(save_path)
        self.connection = self._connect_to_backend()

    def _connect_to_backend(self) -> openeo.Connection:
        """
        Connects and authenticates with the OpenEO it is necessary to be logged in to download data.
        Maybe use github to authenticate, found sometimes I get an error using othwe methods.
        """
        connection = openeo.connect(self.backend_url)
        connection.authenticate_oidc()
        return connection

    def _download_single_product(
        self, 
        product_name: str, 
        tile_id: str, 
        date: str, 
        info: dict, 
        bbox: dict
    ) -> None:
        """
        Downloads a single product-band combination for a specific tile and date.
        """
        print(f"Downloading {product_name} for {tile_id} on {date}...")
        try:
            cube = self.connection.load_collection(
                info["collection"],
                spatial_extent=bbox,
                temporal_extent=[date, date],
                bands=[info["band"]]
            )
            filename = f"{product_name}_{tile_id}_{date}.tiff"
            full_path = os.path.join(self.save_path, filename)
            cube.download(full_path, format="GTiff")
            print(f"Saved as {full_path}")
        except Exception as e:
            print(f"Failed: {product_name} - {tile_id} - {date} → {e}")

    def download_all(self) -> None:
        """
        Iterates over all combinations of products, tiles, and dates to trigger downloads.
        """
        for product_name, info in self.products.items():
            for tile_id, bbox in self.tiles.items():
                for date in self.dates:
                    self._download_single_product(product_name, tile_id, date, info, bbox)
