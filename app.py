from flask import Flask
from flask_cors import CORS
from config.settings import Config
from routes.content_based import content_bp
from routes.collaborative import collaborative_bp
from routes.common import common_bp

def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS
    CORS(app)

    # Register blueprints
    app.register_blueprint(content_bp, url_prefix='/api/content-based')
    app.register_blueprint(collaborative_bp, url_prefix='/api/collaborative')
    app.register_blueprint(common_bp, url_prefix='/api')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
