const TYRADEX_URL = 'https://tyradex.app/api/v1/pokemon/'
const POKEAPI_ESPECE_URL = 'https://pokeapi.co/api/v2/pokemon-species/'
const POKEAPI_POKEMON_URL = 'https://pokeapi.co/api/v2/pokemon/'
const POKEAPI_SPRITE_URL = 'https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/'

function spritePokeapi(pokedexId) {
    return `${POKEAPI_SPRITE_URL}${pokedexId}.png`
}

const GENERATIONS = [
    { id: 'generation-i', jeu: 'red-blue', label: 'Gen I' },
    { id: 'generation-ii', jeu: 'crystal', label: 'Gen II' },
    { id: 'generation-iii', jeu: 'emerald', label: 'Gen III' },
    { id: 'generation-iv', jeu: 'diamond-pearl', label: 'Gen IV' },
    { id: 'generation-v', jeu: 'black-white', label: 'Gen V' },
    { id: 'generation-vi', jeu: 'x-y', label: 'Gen VI' },
    { id: 'generation-vii', jeu: 'ultra-sun-ultra-moon', label: 'Gen VII' },
    { id: 'generation-viii', jeu: 'brilliant-diamond-shining-pearl', label: 'Gen VIII' },
    { id: 'generation-ix', jeu: 'scarlet-violet', label: 'Gen IX' },
]

function construireSpritesDisponibles(spritesPokeapi) {
    const disponibles = [{ id: 'defaut', label: 'defaut', url: spritesPokeapi.front_default }]

    GENERATIONS.forEach((generation) => {
        const url = spritesPokeapi.versions?.[generation.id]?.[generation.jeu]?.front_default

        if (url) {
            disponibles.push({ id: generation.id, label: generation.label, url })
        }
    })

    return disponibles
}

const NOMS_STATS = {
    fr: {
        hp: 'PV',
        atk: 'Attaque',
        def: 'Défense',
        spe_atk: 'Atq. Spé.',
        spe_def: 'Déf. Spé.',
        vit: 'Vitesse',
    },
    en: {
        hp: 'HP',
        atk: 'Attack',
        def: 'Defense',
        spe_atk: 'Sp. Atk',
        spe_def: 'Sp. Def',
        vit: 'Speed',
    },
}

const TYPES_EN = {
    Normal: 'Normal',
    Feu: 'Fire',
    Eau: 'Water',
    Plante: 'Grass',
    Électrik: 'Electric',
    Glace: 'Ice',
    Combat: 'Fighting',
    Poison: 'Poison',
    Sol: 'Ground',
    Vol: 'Flying',
    Psy: 'Psychic',
    Insecte: 'Bug',
    Roche: 'Rock',
    Spectre: 'Ghost',
    Dragon: 'Dragon',
    Ténèbres: 'Dark',
    Acier: 'Steel',
    Fée: 'Fairy',
}

const CORRECTIONS_AUTOREFERENCE = {
    143: 446, // Ronflex : Tyradex renvoie Ronflex comme pré-évolution au lieu de Goinfrex
    446: 143, // Goinfrex : même bug, dans l'autre sens
}

function corrigerEtapeEvolution(etape, idActuel) {
    if (etape.pokedex_id !== idActuel) {
        return etape
    }

    const idCorrige = CORRECTIONS_AUTOREFERENCE[idActuel]
    return idCorrige ? { pokedex_id: idCorrige } : null
}

const TEXTES = {
    fr: {
        labelRecherche: 'Nom (français) ou numéro du Pokémon :',
        placeholderRecherche: 'salamèche ou 4',
        rechercherBouton: 'Rechercher',
        typesLabel: 'Types :',
        voirStats: 'Voir les statistiques',
        masquerStats: 'Masquer les statistiques',
        voirDescription: 'Voir la description',
        masquerDescription: 'Masquer la description',
        vide: 'Veuillez entrer un nom ou un numéro de Pokémon.',
        introuvable: 'Pokémon introuvable.',
        erreurGenerique: 'Une erreur est survenue lors de la recherche.',
        aucuneDescription: 'Aucune description disponible.',
        spriteDefaut: 'Actuel',
        voirRadar: 'Vue radar',
        voirBarres: 'Vue en barres',
        identificationEnCours: "Analyse de l'image en cours...",
        identificationResultat: (confiance) => `Identifié depuis l'image (confiance : ${confiance})`,
        identificationErreur: "Impossible d'identifier ce Pokémon depuis cette image.",
        serveurInjoignable: "Serveur injoignable : lancez « node server.js » puis ouvrez http://localhost:3000.",
        imageTropGrosse: 'Image trop volumineuse (10 Mo maximum).',
    },
    en: {
        labelRecherche: 'Name (English) or Pokédex number:',
        placeholderRecherche: 'charmander or 4',
        rechercherBouton: 'Search',
        typesLabel: 'Types:',
        voirStats: 'Show stats',
        masquerStats: 'Hide stats',
        voirDescription: 'Show description',
        masquerDescription: 'Hide description',
        vide: 'Please enter a Pokémon name or number.',
        introuvable: 'Pokémon not found.',
        erreurGenerique: 'An error occurred during the search.',
        aucuneDescription: 'No description available.',
        spriteDefaut: 'Current',
        voirRadar: 'Radar view',
        voirBarres: 'Bar view',
        identificationEnCours: 'Analyzing image...',
        identificationResultat: (confiance) => `Identified from image (confidence: ${confiance})`,
        identificationErreur: 'Could not identify a Pokémon from this image.',
        serveurInjoignable: 'Server unreachable: run "node server.js" and open http://localhost:3000.',
        imageTropGrosse: 'Image too large (10 MB max).',
    },
}

const inputRecherche = document.getElementById('recherche')
const labelRecherche = document.getElementById('label-recherche')
const boutonChercher = document.getElementById('chercher')
const boutonLangue = document.getElementById('bouton-langue')
const boutonOeil = document.getElementById('bouton-oeil')
const inputImage = document.getElementById('input-image')
const statutIdentification = document.getElementById('statut-identification')
const apercuIdentification = document.getElementById('apercu-identification')
const imageIdentification = document.getElementById('image-identification')
const erreur = document.getElementById('erreur')
const resultat = document.getElementById('resultat')

let langueActuelle = 'fr'
let dernieresDonnees = null
let spriteChoisieId = 'defaut'
let graphiqueRadar = null
let modeStats = 'radar'

function appliquerTextesStatiques() {
    const textes = TEXTES[langueActuelle]

    labelRecherche.textContent = textes.labelRecherche
    inputRecherche.placeholder = textes.placeholderRecherche
    boutonChercher.textContent = textes.rechercherBouton
    boutonLangue.textContent = langueActuelle === 'fr' ? 'EN' : 'FR'
}

function traduireType(nomFr, langue) {
    return langue === 'fr' ? nomFr : (TYPES_EN[nomFr] || nomFr)
}

function trouverDescription(espece, langue) {
    const entree = espece.flavor_text_entries.find((e) => e.language.name === langue)
        || espece.flavor_text_entries.find((e) => e.language.name === 'fr')
        || espece.flavor_text_entries.find((e) => e.language.name === 'en')

    if (!entree) {
        return TEXTES[langue].aucuneDescription
    }

    return entree.flavor_text.replace(/[\f\n\r]+/g, ' ')
}

function construireBarresStats(pokemon, langue) {
    const conteneur = document.createElement('div')
    conteneur.id = 'barres-stats'

    Object.entries(pokemon.stats).forEach(([cle, valeur]) => {
        const ligne = document.createElement('div')
        ligne.className = 'stat-ligne'

        const nom = document.createElement('span')
        nom.className = 'stat-nom'
        nom.textContent = NOMS_STATS[langue][cle] || cle

        const barre = document.createElement('div')
        barre.className = 'stat-barre'

        const remplissage = document.createElement('div')
        remplissage.className = 'stat-barre-remplissage'
        remplissage.style.width = `${Math.min(valeur, 200) / 2}%`
        barre.append(remplissage)

        const valeurElement = document.createElement('span')
        valeurElement.className = 'stat-valeur'
        valeurElement.textContent = valeur

        ligne.append(nom, barre, valeurElement)
        conteneur.append(ligne)
    })

    return conteneur
}

function construireStats(textes) {
    const conteneur = document.createElement('div')
    conteneur.id = 'stats-pokemon'
    conteneur.hidden = true

    const enteteStats = document.createElement('div')
    enteteStats.id = 'entete-stats'

    const boutonMode = document.createElement('button')
    boutonMode.id = 'bouton-mode-stats'
    boutonMode.type = 'button'
    boutonMode.className = 'bouton-sprite'
    boutonMode.textContent = modeStats === 'radar' ? textes.voirBarres : textes.voirRadar

    boutonMode.addEventListener('click', () => {
        modeStats = modeStats === 'radar' ? 'barres' : 'radar'
        afficherResultat()
    })

    enteteStats.append(boutonMode)

    const contenuStats = document.createElement('div')
    contenuStats.id = 'contenu-stats'

    conteneur.append(enteteStats, contenuStats)

    return conteneur
}

function creerGraphiqueRadar(canvas, pokemon, langue) {
    const labels = Object.keys(pokemon.stats).map((cle) => NOMS_STATS[langue][cle] || cle)
    const valeurs = Object.values(pokemon.stats)

    return new Chart(canvas, {
        type: 'radar',
        data: {
            labels,
            datasets: [{
                label: pokemon.name[langue],
                data: valeurs,
                backgroundColor: 'rgba(255, 204, 0, 0.25)',
                borderColor: '#ffcc00',
                pointBackgroundColor: '#ffcc00',
            }],
        },
        options: {
            aspectRatio: 1,
            scales: {
                r: {
                    beginAtZero: true,
                    suggestedMax: 150,
                    ticks: { display: false },
                    grid: { color: 'rgba(214, 255, 214, 0.2)' },
                    angleLines: { color: 'rgba(214, 255, 214, 0.2)' },
                    pointLabels: { color: '#d6ffd6', font: { size: 10 } },
                },
            },
            plugins: {
                legend: { display: false },
            },
        },
    })
}

function afficherResultat() {
    if (!dernieresDonnees) {
        return
    }

    const { pokemon, espece, evolutionsPokemon, spritesDisponibles } = dernieresDonnees
    const langue = langueActuelle
    const textes = TEXTES[langue]

    const statsPrecedentes = document.getElementById('stats-pokemon')
    const descriptionPrecedente = document.getElementById('description-pokemon')
    const statsVisibles = statsPrecedentes ? !statsPrecedentes.hidden : false
    const descriptionVisible = descriptionPrecedente ? !descriptionPrecedente.hidden : false

    if (graphiqueRadar) {
        graphiqueRadar.destroy()
        graphiqueRadar = null
    }

    resultat.replaceChildren()

    const spriteActuel = spritesDisponibles.find((s) => s.id === spriteChoisieId) || spritesDisponibles[0]

    const image = document.createElement('img')
    image.className = 'sprite-principal'
    image.src = spriteActuel.url
    image.alt = pokemon.name[langue]

    const selecteurSprites = document.createElement('div')
    selecteurSprites.id = 'selecteur-sprites'

    spritesDisponibles.forEach((option) => {
        const boutonSprite = document.createElement('button')
        boutonSprite.type = 'button'
        boutonSprite.className = 'bouton-sprite' + (option.id === spriteActuel.id ? ' actif' : '')
        boutonSprite.textContent = option.id === 'defaut' ? textes.spriteDefaut : option.label

        boutonSprite.addEventListener('click', () => {
            spriteChoisieId = option.id
            afficherResultat()
        })

        selecteurSprites.append(boutonSprite)
    })

    const titre = document.createElement('h2')
    titre.className = 'titre-pokemon'
    titre.textContent = `#${pokemon.pokedex_id} - ${pokemon.name[langue]}`

    const types = document.createElement('p')
    types.className = 'types-pokemon'
    types.textContent = `${textes.typesLabel} ${pokemon.types.map((t) => traduireType(t.name, langue)).join(', ')}`

    const boutonStats = document.createElement('button')
    boutonStats.id = 'bouton-stats'
    boutonStats.textContent = textes.voirStats

    const statsPokemon = construireStats(textes)
    statsPokemon.hidden = !statsVisibles
    boutonStats.textContent = statsPokemon.hidden ? textes.voirStats : textes.masquerStats

    const contenuStats = statsPokemon.querySelector('#contenu-stats')
    let canvasStats = null

    if (modeStats === 'radar') {
        canvasStats = document.createElement('canvas')
        canvasStats.id = 'graphique-stats'
        contenuStats.append(canvasStats)
    } else {
        contenuStats.append(construireBarresStats(pokemon, langue))
    }

    boutonStats.addEventListener('click', () => {
        statsPokemon.hidden = !statsPokemon.hidden
        boutonStats.textContent = statsPokemon.hidden ? textes.voirStats : textes.masquerStats

        if (!statsPokemon.hidden && modeStats === 'radar' && !graphiqueRadar) {
            graphiqueRadar = creerGraphiqueRadar(canvasStats, pokemon, langue)
        }
    })

    const boutonDescription = document.createElement('button')
    boutonDescription.id = 'bouton-description'
    boutonDescription.textContent = textes.voirDescription

    const descriptionPokemon = document.createElement('p')
    descriptionPokemon.id = 'description-pokemon'
    descriptionPokemon.textContent = trouverDescription(espece, langue)
    descriptionPokemon.hidden = !descriptionVisible
    boutonDescription.textContent = descriptionPokemon.hidden ? textes.voirDescription : textes.masquerDescription

    boutonDescription.addEventListener('click', () => {
        descriptionPokemon.hidden = !descriptionPokemon.hidden
        boutonDescription.textContent = descriptionPokemon.hidden ? textes.voirDescription : textes.masquerDescription
    })

    const ligneEvolution = document.createElement('div')
    ligneEvolution.id = 'ligne-evolution'

    evolutionsPokemon.forEach((evoPokemon) => {
        const etape = document.createElement('div')
        etape.className = 'evolution-etape'

        const imageEvo = document.createElement('img')
        imageEvo.className = 'evolution-image'
        imageEvo.src = spritePokeapi(evoPokemon.pokedex_id)
        imageEvo.alt = evoPokemon.name[langue]
        imageEvo.width = 60

        const boutonEvo = document.createElement('button')
        boutonEvo.type = 'button'
        boutonEvo.className = 'evolution-bouton'
        boutonEvo.title = evoPokemon.name[langue]
        boutonEvo.append(imageEvo)

        boutonEvo.addEventListener('click', () => {
            inputRecherche.value = String(evoPokemon.pokedex_id)
            chercherPokemon()
        })

        const nomEvo = document.createElement('p')
        nomEvo.textContent = evoPokemon.name[langue]

        etape.append(boutonEvo, nomEvo)
        ligneEvolution.append(etape)
    })

    resultat.append(image, selecteurSprites, titre, types, boutonStats, statsPokemon, boutonDescription, descriptionPokemon, ligneEvolution)

    if (!statsPokemon.hidden && modeStats === 'radar') {
        graphiqueRadar = creerGraphiqueRadar(canvasStats, pokemon, langue)
    }
}

// Also called directly as an event listener, where the argument is the event
// (no garderApercu property): any manual search hides the uploaded image.
async function chercherPokemon({ garderApercu = false } = {}) {
    const nomOuId = inputRecherche.value.trim().toLowerCase()
    const textes = TEXTES[langueActuelle]

    if (!garderApercu) {
        masquerApercu()
    }

    erreur.textContent = ''
    statutIdentification.textContent = ''
    resultat.replaceChildren()
    dernieresDonnees = null
    spriteChoisieId = 'defaut'

    if (nomOuId === '') {
        erreur.textContent = textes.vide
        return
    }

    try {
        const response = await fetch(`${TYRADEX_URL}${encodeURIComponent(nomOuId)}`)
        const pokemon = await response.json()

        if (!response.ok || !pokemon.pokedex_id) {
            erreur.textContent = textes.introuvable
            return
        }

        const [especeResponse, pokeapiResponse] = await Promise.all([
            fetch(`${POKEAPI_ESPECE_URL}${pokemon.pokedex_id}/`),
            fetch(`${POKEAPI_POKEMON_URL}${pokemon.pokedex_id}/`),
        ])
        const espece = await especeResponse.json()
        const pokeapiData = await pokeapiResponse.json()
        const spritesDisponibles = construireSpritesDisponibles(pokeapiData.sprites)

        const etapesEvolution = [
            ...(pokemon.evolution?.pre ?? []).map((etape) => corrigerEtapeEvolution(etape, pokemon.pokedex_id)),
            { pokedex_id: pokemon.pokedex_id },
            ...(pokemon.evolution?.next ?? []).map((etape) => corrigerEtapeEvolution(etape, pokemon.pokedex_id)),
        ].filter(Boolean)

        const evolutionsPokemon = await Promise.all(
            etapesEvolution.map((etape) => fetch(`${TYRADEX_URL}${etape.pokedex_id}`).then((r) => r.json()))
        )

        dernieresDonnees = { pokemon, espece, evolutionsPokemon, spritesDisponibles }
        afficherResultat()
    } catch (error) {
        console.error(error)
        erreur.textContent = textes.erreurGenerique
    }
}

boutonChercher.addEventListener('click', chercherPokemon)

inputRecherche.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        chercherPokemon()
    }
})

boutonLangue.addEventListener('click', () => {
    langueActuelle = langueActuelle === 'fr' ? 'en' : 'fr'
    appliquerTextesStatiques()
    afficherResultat()
})

const TAILLE_MAX_IMAGE = 10 * 1024 * 1024 // same limit as MAX_UPLOAD_SIZE in server.js

function afficherApercu(fichier) {
    masquerApercu()
    imageIdentification.src = URL.createObjectURL(fichier)
    apercuIdentification.hidden = false
    apercuIdentification.classList.add('analyse-en-cours')
}

function masquerApercu() {
    if (imageIdentification.src) {
        URL.revokeObjectURL(imageIdentification.src)
        imageIdentification.removeAttribute('src')
    }
    apercuIdentification.hidden = true
    apercuIdentification.classList.remove('analyse-en-cours')
}

async function identifierPokemon(fichier) {
    afficherApercu(fichier)
    try {
        await envoyerPourIdentification(fichier)
    } finally {
        // The image stays displayed (even on failure, to show what was sent);
        // only the "analysis in progress" effect stops.
        apercuIdentification.classList.remove('analyse-en-cours')
    }
}

async function envoyerPourIdentification(fichier) {
    const textes = TEXTES[langueActuelle]

    erreur.textContent = ''
    statutIdentification.textContent = textes.identificationEnCours

    // Checked here too: server.js aborts oversized uploads mid-stream, which
    // the browser reports as a network error rather than a 413
    if (fichier.size > TAILLE_MAX_IMAGE) {
        statutIdentification.textContent = ''
        erreur.textContent = textes.imageTropGrosse
        return
    }

    let response
    try {
        response = await fetch('/api/identifier', {
            method: 'POST',
            headers: { 'Content-Type': fichier.type || 'application/octet-stream' },
            body: fichier,
        })
    } catch (error) {
        // fetch only rejects when no HTTP response comes back at all: server
        // not started, or the page opened outside it (file://, editor preview)
        console.error(error)
        statutIdentification.textContent = ''
        erreur.textContent = textes.serveurInjoignable
        return
    }

    try {
        const identification = await response.json()

        if (!response.ok || !identification.id) {
            statutIdentification.textContent = ''
            erreur.textContent = response.status === 413 ? textes.imageTropGrosse : textes.identificationErreur
            return
        }

        inputRecherche.value = String(identification.id)
        apercuIdentification.classList.remove('analyse-en-cours')
        await chercherPokemon({ garderApercu: true })
        statutIdentification.textContent = textes.identificationResultat(`${identification.name} — ${(identification.confidence * 100).toFixed(1)}%`)
    } catch (error) {
        console.error(error)
        statutIdentification.textContent = ''
        erreur.textContent = textes.identificationErreur
    }
}

boutonOeil.addEventListener('click', () => {
    inputImage.click()
})

inputImage.addEventListener('change', () => {
    const fichier = inputImage.files[0]
    inputImage.value = ''

    if (fichier) {
        identifierPokemon(fichier)
    }
})
