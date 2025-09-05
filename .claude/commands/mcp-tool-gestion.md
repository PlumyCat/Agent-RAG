# mcp-tools

Workflow de création et modification d'outils pour serveur MCP Python.

## Process de développement d'outil MCP

1. **Analyser** le besoin et définir l'interface
2. **Créer** le schéma JSON de l'outil  
3. **Implémenter** le handler avec validation
4. **Tester** avec l'inspector MCP
5. **Intégrer** au serveur principal
6. **Documenter** usage et exemples

## Template d'outil MCP

### Schéma d'outil (dans tools.py)
```python
from mcp.types import Tool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    param1: str = Field(description="Description du paramètre 1")
    param2: int = Field(default=10, description="Paramètre optionnel")

MY_TOOL = Tool(
    name="my_tool",
    description="Description claire de ce que fait l'outil",
    inputSchema=MyToolInput.model_json_schema()
)
```

### Handler d'outil (dans handlers.py)
```python
from mcp.types import TextContent
import logging

logger = logging.getLogger("mcp.server")

async def handle_my_tool(arguments: dict) -> list[TextContent]:
    """Handler pour mon outil MCP."""
    try:
        # Validation des inputs
        validated_input = MyToolInput(**arguments)
        
        # Logique métier
        logger.debug(f"Executing my_tool with: {validated_input}")
        result = process_data(validated_input.param1, validated_input.param2)
        
        # Retour conforme MCP
        return [TextContent(
            type="text", 
            text=f"Résultat: {result}"
        )]
        
    except ValueError as e:
        logger.error(f"Invalid input for my_tool: {e}")
        raise ValueError(f"Erreur de paramètre: {e}")
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        raise RuntimeError(f"Échec de l'outil: {e}")

def process_data(param1: str, param2: int) -> str:
    """Logique métier séparée pour faciliter les tests."""
    # Implémentation...
    return f"Processed {param1} with {param2}"
```

### Tests d'outil (dans tests/)
```python
import pytest
from your_mcp_server.handlers import handle_my_tool
from your_mcp_server.tools import MyToolInput

@pytest.mark.asyncio
class TestMyTool:
    
    async def test_valid_input(self):
        """Test avec inputs valides."""
        result = await handle_my_tool({
            "param1": "test", 
            "param2": 5
        })
        assert len(result) == 1
        assert "Processed test with 5" in result[0].text
    
    async def test_invalid_input(self):
        """Test validation des inputs."""
        with pytest.raises(ValueError):
            await handle_my_tool({"param1": ""})  # param1 requis
    
    async def test_default_values(self):
        """Test valeurs par défaut."""
        result = await handle_my_tool({"param1": "test"})
        assert "with 10" in result[0].text  # default param2=10
```

## Bonnes pratiques MCP

### Validation robuste
- Utiliser Pydantic pour validation inputs
- Vérifier types et contraintes métier  
- Messages d'erreur clairs et actionables

### Gestion d'erreurs
- Catch exceptions spécifiques
- Log pour debugging sans exposer détails
- Retours d'erreur conformes au protocole MCP

### Performance
- Opérations async quand possible
- Éviter les opérations bloquantes longues
- Timeout sur opérations I/O

### Tests
- Test unitaire par outil
- Test cas limites et erreurs  
- Test intégration avec inspector MCP
- Mock des dépendances externes

## Commandes de validation

```bash
# Test outil spécifique
pytest tests/test_my_tool.py -v

# Test avec inspector MCP
npx -y @modelcontextprotocol/inspector python -m your_server

# Validation complète
python -m your_server --validate-tools
```

Utilise cette structure pour créer des outils MCP robustes et maintenables.