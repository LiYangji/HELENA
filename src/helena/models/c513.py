"""Validated blob-comoving C513 scenario and response preparation.

Importing this canonical model intentionally enables global JAX x64 before any
scientific arrays are built; backend kernels also fail fast when x64 is absent.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from helena.core.gamma_gamma import _MEC2_CGS
from helena.core.jax_backend.chains.steady_state_hadronic_cascade import (
    _target_components,
    solve_steady_state_hadronic_cascade,
)
from helena.core.jax_backend.kernels.gamma_gamma_pair_injection import (
    prepare_gamma_gamma_pair_response_jax,
)
from helena.core.jax_backend.radiation.inverse_compton import inverse_compton_sed_jax
from helena.core.jax_backend.radiation.ssa import _transfer_from_quantities
from helena.core.jax_backend.radiation.synchrotron import electron_synchrotron_sed_jax
from helena.core.jax_backend.transport.electron_positron_transport import (
    electron_positron_steady_state_transport_jax,
)
from helena.core.observer import (
    sed_luminosity_to_energy_flux,
    transform_comoving_sed,
)
from helena.core.photomeson import _PROTON_REST_ENERGY_CGS
from helena.core.propagation import with_observer_propagation
from helena.core.synchrotron import _C_CGS
from helena.runtime_cache import (
    bethe_heitler_response_fingerprint,
    load_or_prepare_bethe_heitler_pair_response_jax,
)

jax.config.update("jax_enable_x64", True)

__all__ = [
    "build_c513_configuration",
    "c513_configuration_from_escape_rate_offset",
    "c513_configuration_from_injection_shape_offsets",
    "c513_configuration_from_log_offsets",
    "prepare_c513_responses",
    "run_c513_forward",
]

SCENARIO_NAME = "c513_preflight"
REFERENCE = (
    "Cerruti et al., A Comprehensive Hadronic Code Comparison for Active "
    "Galactic Nuclei, arXiv:2411.14218v3, Table 3 and Section 4"
)
REFERENCE_URL = "https://arxiv.org/abs/2411.14218"

EV_TO_ERG = 1.602176634e-12
RADIUS_CM = 1.0e16
MAGNETIC_FIELD_G = 0.1
DOPPLER_FACTOR = 30.0
ELECTRON_CUTOFF_GAMMA = 3.0e5
PROTON_CUTOFF_GAMMA = 1.0e7
ELECTRON_KINETIC_INJECTION_LUMINOSITY_ERG_S = 3.7e40
PROTON_KINETIC_INJECTION_LUMINOSITY_ERG_S = 2.8e46
OUTER_RTOL = 1.0e-3
MAX_OUTER_ITERATIONS = 30
REQUIRED_CONSECUTIVE_PASSES = 2
INNER_RTOL = 1.0e-4
INNER_MAXIMUM_ITERATIONS = 32

_INJECTION_QUADRATURE_NODES, _INJECTION_QUADRATURE_WEIGHTS = (
    np.polynomial.legendre.leggauss(64)
)

# Canonical float64 bits for the validated C513 daughter grid.
_C513_DAUGHTER_GRID_BITS = (
    0x3ff000010c6f7a0b, 0x3ff0fc4f5c5e21bd, 0x3ff208284f9d357f, 0x3ff32480f9ba58c5,
    0x3ff4525d86e5b033, 0x3ff592d229ffb4ff, 0x3ff6e7041950ee3d, 0x3ff8502a9ad2cbbc,
    0x3ff9cf9020ff1c6d, 0x3ffb6693792ab924, 0x3ffd16a90c800a05, 0x3ffee15c34bf176c,
    0x4000642852fd771e, 0x400166a1f74faa23, 0x40027907878f1640, 0x40039c54155cc7fa,
    0x4004d192297b4d1b, 0x400619dcb7aea675, 0x4007766021a1e967, 0x4008e85b49bf7438,
    0x400a7120b6f72fc9, 0x400c1217ca7dd6bb, 0x400dccbe08a0ac02, 0x400fa2a875da801e,
    0x4010cac284b4b647, 0x4011d38e1b5c2ce6, 0x4012eca9478223b9, 0x401417153e5dbced,
    0x401553e30d161d9b, 0x4016a4349298ed6c, 0x4018093d88d48e72, 0x401984449e48bb02,
    0x401b16a4a0ef2a3c, 0x401cc1cdbb8dba91, 0x401e8746c6947a35, 0x4020345756dde3ff,
    0x402133def5d4e1e4, 0x402243240e7d3696, 0x4023631ed60d9a7c, 0x402494d6cbc0bec3,
    0x4025d963a9edd142, 0x402731ee65faddb7, 0x40289fb240133c06, 0x402a23fde39aa8ad,
    0x402bc0349964f4a7, 0x402d75cf8cca7bc6, 0x402f465f24c2d34a, 0x403099c63920ba35,
    0x40319f8d5a91ccf3, 0x4032b5747c89971b, 0x4033dc79e5ec4983, 0x403515ab87562818,
    0x40366227f2193691, 0x4037c31f5e71b840, 0x403939d4c1e16d29, 0x403ac79ef6b03d71,
    0x403c6de9f5a0b455, 0x403e2e3822f75405, 0x40400511d802bcc7, 0x404101b0093da694,
    0x40420dddc96c1457, 0x40432a9079b3027b, 0x404458cc98a179c2, 0x404599a6b089c0de,
    0x4046ee4454891870, 0x404857dd2d258637, 0x4049d7bc1577773c, 0x404b6f4049e42455,
    0x404d1fdea97db6ae, 0x404eeb230b2f3a64, 0x40506958d3f644e6, 0x40516c244d98cb47,
    0x40527ee0bd987007, 0x4053a289851348e3, 0x4054d829812e62e9, 0x405620dbff42ea2c,
    0x40577dcdc015c2dd, 0x4058f03e0b06c198, 0x405a797fd2334d5a, 0x405c1afae897b78b,
    0x405dd62d4b4b0de2, 0x405facac7f02a494, 0x4060d013818e92b4, 0x4061d932edc451f4,
    0x4062f2a719760fa7, 0x40641d718b4987a0, 0x40655aa3a6d894db, 0x4066ab5fa6d6ca60,
    0x406810d9a69fa396, 0x40698c58bc2046fd, 0x406b1f38231ecd20, 0x406ccae87af0e1b1,
    0x406e90f117c47328, 0x40703978b457862e, 0x4071395139e5bb77, 0x407248ec34e0b9e3,
    0x407369422914611b, 0x40749b5ae927b6ef, 0x4075e04e8801c024, 0x40773946590ce9e9,
    0x4078a77e00437e6a, 0x407a2c4492fe0fe0, 0x407bc8fdca8c1b5d, 0x407d7f2349af6ac7,
    0x407f5045f62405d7, 0x40809f07b3b86d26, 0x4081a521b6285e05, 0x4082bb60d40c05b5,
    0x4083e2c3a4c8a13f, 0x40851c586e70fb07, 0x4086693e1d1149d3, 0x4087caa54936b477,
    0x408941d14ea2af87, 0x408ad019742936f8, 0x408c76ea25c89b0e, 0x408e37c6421a43ea,
    0x40900a243e26c1e6, 0x4091071269f82d9e, 0x4092139511ef0256, 0x409330a1e4e02619,
    0x40945f3db3d399ef, 0x4095a07d60a9216a, 0x4096f586db701a8d, 0x40985f922f5a546d,
    0x4099dfeaa03ff99c, 0x409b77efd9b9cc87, 0x409d291730e70b9e, 0x409ef4ecfa0567a5,
    0x40a06e8af989ccfb, 0x40a171a8626534a0, 0x40a284bbcda62d63, 0x40a3a8c0ec0100ab,
    0x40a4dec2ef17f228, 0x40a627dd7df5b5e5, 0x40a7853db8971aaa, 0x40a8f8234b716224,
    0x40aa81e193f1602e, 0x40ac23e0d6ff02b5, 0x40addf9f8aa16008, 0x40afb6b3b3f0e8ec,
    0x40b0d5662d4bf84c, 0x40b1ded9899abb1e, 0x40b2f8a6d105856d, 0x40b423cfdbba74c1,
    0x40b5616663e1788e, 0x40b6b28d00111d5e, 0x40b818782d30d056, 0x40b9946f68bbe2de,
    0x40bb27ce5c6791fb, 0x40bcd4061c3e3dfd, 0x40be9a9e7852e2d3, 0x40c03e9bb1a0e723,
    0x40c13ec537634d52, 0x40c24eb62fe1d498, 0x40c36f676d9664ef, 0x40c4a1e116aec23e,
    0x40c5e73b96bdd22d, 0x40c740a09f50274e, 0x40c8af4c384e9785, 0x40ca348de1281692,
    0x40cbd1c9c3cb6a51, 0x40cd8879fa8998ed, 0x40cf5a2fea0b3f3d, 0x40d0a44ad84aad58,
    0x40d1aab7d5f6c9bf, 0x40d2c14f0ba1623a, 0x40d3e90f614a3cd8, 0x40d523077295b91b,
    0x40d670568666fa72, 0x40d7d22d95ba896b, 0x40d949d062b1ee5a, 0x40dad896a0cf96e1,
    0x40dc7fed2f7213da, 0x40de415767ae6e2f, 0x40e00f383f5d85a0, 0x40e10c767f17b4d6,
    0x40e2194e29b87d69, 0x40e336b53bdd47a0, 0x40e465b0d921291a, 0x40e5a7563b0d1a56,
    0x40e6fccbaec0041c, 0x40e86749a236bbf0, 0x40e9e81bc22a53c6, 0x40eb80a2298a4c97,
    0x40ed3252a3a85a04, 0x40eefeba023c7ec2, 0x40f073bec43d393a, 0x40f1772e364243f0,
    0x40f28a98b84e613e, 0x40f3aefa4ac540c8, 0x40f4e55e73e11ca6, 0x40f62ee1347a9666,
    0x40f78cb00be48d01, 0x40f9000b0bc9afd7, 0x40fa8a45fd083911, 0x40fc2cc99697c470,
    0x40fde914c795bac6, 0x40ffc0be15a64f17, 0x4100daba88755267, 0x4101e481ef703afc,
    0x4102fea86eca4359, 0x41042a303053bb03, 0x4105682b44de0d1f, 0x4106b9bc9effd715,
    0x410820191d4b59f1, 0x41099c88a4eadb2b, 0x410b30674da58a3f, 0x410cdd26a05f6f2d,
    0x410ea44ee937ccf2, 0x411043c04f3dac37, 0x4111443aeed958bd, 0x411254820014e3d4,
    0x4113758ea43126a7, 0x4114a86954fd151a, 0x4115ee2ad6d3884a, 0x411747fd398105e7,
    0x4118b71ce8fc92e4, 0x411a3cd9ceed1ae8, 0x411bda988604547c, 0x411d91d3a0485c6a,
    0x411f641d01769397, 0x4120a98fa75e5823, 0x4121b04fba8c3c35, 0x4122c73f23e1aa7d,
    0x4123ef5d1c1276e1, 0x412529b8946fad64, 0x412677712ed02077, 0x4127d9b844be42a3,
    0x412951d1fedc1966, 0x412ae1167d7ceca8, 0x412c88f3138414eb, 0x412e4aeb94a90331,
    0x4130144ddc292d37, 0x413111dc492665c6, 0x41321f09115b31db, 0x41333cca7f461c49,
    0x41346c26092f73d8, 0x4135ae31406526ea, 0x41370412cf331f6b, 0x41386f03868080ad,
    0x4139f04f7c087850, 0x413b89573a3485d5, 0x413d3b9102ae3dcc, 0x413f088a24cfaedc,
    0x414078f43495dd91, 0x41417cb5c9bd8396, 0x414290777e274e00, 0x4143b535a1ff8d20,
    0x4144ebfc103339ad, 0x414635e723855167, 0x41479424babcf248, 0x414907f54cda4489,
    0x414a92ad0e4eecc5, 0x414c35b528465134, 0x414df28d031a8323, 0x414fcacba5242a85,
    0x4150e010939337a3, 0x4151ea2c1fd5d1ef, 0x415304abf35e3862, 0x4154309289b8c4c1,
    0x41556ef24a7bcde0, 0x4156c0ee845b228b, 0x415827bc77b2c397, 0x4159a4a4717cbdf8,
    0x415b3902f7b50d03, 0x415ce64a083e5ee8, 0x415eae026b6b83d7, 0x416048e68db1a482,
    0x416149b260d3cb53, 0x41625a4fa60e739f, 0x41637bb7cd8258c9, 0x4164aef3a4ba18b6,
    0x4165f51c48f49bcd, 0x41674f5c285c318e, 0x4168bef01315bb8a, 0x416a45285d21be29,
    0x416be36a12189421, 0x416d9b303bdb56fd, 0x416f6e0d3d6467b5, 0x4170aed6217a75b5,
    0x4171b5e964780ee8, 0x4172cd311d650c5c, 0x4173f5acd5c2dcf4, 0x4175306bd4aa5901,
    0x41767e8e1702cdad, 0x4177e145570328fd, 0x417959d623ee6170, 0x417ae9990b0b0c8b,
    0x417c91fbd2e5dd7a, 0x417e5482c9ff80ad, 0x41801965150c0747, 0x41811743c8ae95db,
    0x418224c5c969fa7b, 0x418342e1afb68a87, 0x4184729d44a3fb05, 0x4185b50e7160f9d0,
    0x41870b5c3d83f1f2, 0x418876bfdcfda561, 0x4189f885ceac9c64, 0x418b920f0c97a007,
    0x418d44d24ee59daf, 0x418f125d62ba765f, 0x41907e2b4b193844, 0x4191823f1d64aaeb,
    0x419296581fc765b6, 0x4193bb72f24f9c02, 0x4194f29bc4b7d612, 0x41963cef4bc9e55f,
    0x41979b9bc5df5f84, 0x41990fe20f6dfa62, 0x419a9b16c89cd438, 0x419c3ea38cef45b3,
    0x419dfc083e226ad1, 0x419fd4dc636c1ff2, 0x41a0e5684f2e69c8, 0x41a1efd81b5cae63,
    0x41a30ab15f5b8450, 0x41a436f6e88d302a, 0x41a575bb75686cf8, 0x41a6c822b0db64e5,
    0x41a82f623d2ace2d, 0x41a9acc2cf415b2f, 0x41ab41a15b72b72d, 0x41acef7054c540f3,
    0x41aeb7b8ffe6a85f, 0x41b04e0e6d80c8c3, 0x41b14f2b8ddebee4, 0x41b2601f22633f2b,
    0x41b381e2ea27dfdd, 0x41b4b580068d6b73, 0x41b5fc0fedd2fe19, 0x41b756bd6c9e91a8,
    0x41b8c6c5b7629bb2, 0x41ba4d798c9ae512, 0x41bbec3e68ea2b22, 0x41bda48fce327676,
    0x41bf78009ed370b9, 0x41c0b41e472638e2, 0x41c1bb84d449c8a1, 0x41c2d324f8c3e60a,
    0x41c3fbfe8efd2ff7, 0x41c5372133f1738c, 0x41c685ad3fb54d50, 0x41c7e8d4cd4ac252,
    0x41c961dcd2b637fd, 0x41caf21e4a540fea, 0x41cc9b076e7ef61f, 0x41ce5e1d08a7b24f,
    0x41d01e7dea888b90, 0x41d11cacfe3ac633, 0x41d22a845277e07c, 0x41d348facdcaaa4e,
    0x41d479168c247409, 0x41d5bbedceb07d84, 0x41d712a7fa6d3c02, 0x41d87e7ea6746b45,
    0x41da00bebae93789, 0x41db9ac9a19309bf, 0x41dd4e16893bab94, 0x41df1c33bcf8a38a,
    0x41e08364084cf1c2, 0x41e187ca31c59e05, 0x41e29c3a9dc549f7, 0x41e3c1b23c555678,
    0x41e4f93d9218b483, 0x41e643f9adfc89f5, 0x41e7a3152e0b2602, 0x41e917d1544febc3,
    0x41eaa3832cc98c8a, 0x41ec4794c57786e7, 0x41ee058679a07041, 0x41efdef05180259b,
    0x41f0eac1bbcfd5ee, 0x41f1f585e2962c9d, 0x41f310b8b35c779d, 0x41f43d5d4d74cf1d,
    0x41f57c86c651d3b4, 0x41f6cf5925393ddf, 0x41f8370a6e7778ca, 0x41f9b4e3bf08c48a,
    0x41fb4a4279bb6b86, 0x41fcf89986de9344, 0x41fec172a7a22a04, 0x42005337ef2f3b73,
    0x420154a6768679a6, 0x420265f075a830a2, 0x4203880ffabfd24f, 0x4204bc0e7b1ee0a4,
    0x42060305c620d919, 0x42075e210705499b, 0x4208ce9dd6abfd4f, 0x420a55cd5e2db7c9,
    0x420bf5158b5b62ef, 0x420dadf2583df49e, 0x420f81f726c2b459, 0x4210b96818e8ff41,
    0x4211c1220a911d8e, 0x4212d91ab696c5d4, 0x42140252486363e5, 0x42153dd8b2f0eadd,
    0x42168ccea99e2487, 0x4217f066a856d1f6, 0x421969e60c014fda, 0x421afaa63c32552f,
    0x421ca415e73730c1, 0x421e67ba5197b196, 0x422023985d215b17, 0x42212217ea55a3b8,
    0x42223044ad181b94, 0x42234f15da1ec4dd, 0x42247f91e056c8a1, 0x4225c2cf5903d411,
    0x422719f606a9f942, 0x4228863fe3ab528d, 0x422a08fa4191041f, 0x422ba386fa067850,
    0x422d575db29de47a, 0x422f260d34865456, 0x4230889e6cb6dcbd, 0x42318d57076e6ddd,
    0x4232a21ef8b7cbfe, 0x4233c7f380b0d80a, 0x4234ffe178ffcd48, 0x42364b064ad1afb5,
    0x4237aa90f3ffd3d9, 0x42391fc31c4b733d, 0x423aabf23bacf703, 0x423c5088d2c442a4,
    0x423e0f07b687df39, 0x423fe907706283d4, 0x4240f01cda009479, 0x4241fb357613d6de,
    0x424316c1effb9392, 0x424443c5b913a763, 0x424583543de6224d, 0x4246d691e22d8783,
    0x42483eb50c5d0067, 0x4249bd0741a34d83, 0x424b52e6536c52a7, 0x424d01c59f751dec,
    0x424ecb2f6397477b, 0x425058631341491e, 0x42515a231b576e21, 0x42526bc3a0726149,
    0x42538e3effe8788a, 0x4254c29f031680c7, 0x425609fdd2908efb, 0x42576586f84db887,
    0x4258d67871bae9b8, 0x425a5e23d2afa1d1, 0x425bfdef7a4ecc7c, 0x425db757daee5727,
    0x425f8bf0d6318976, 0x4260beb3974a517e, 0x4261c6c107ddef9c, 0x4262df1257766a4d,
    0x426408a802979ff3, 0x426544925254e329, 0x426693f25574121b, 0x4267f7fae8e9586c,
    0x426971f1d09d9cc8, 0x426b0330e1807fc9, 0x426cad273df6a888, 0x426e715aa5c5e5c5,
    0x427028b46d594075, 0x427127848d8a077b, 0x42723606d9de1254, 0x42735532d54f54e2,
    0x4274860f41e11704, 0x4275c9b3110b5736, 0x4277214662f56064, 0x42788e0395691a1c,
    0x427a11386376ff14, 0x427bac4716d1e7bd, 0x427d60a7cbfa106d, 0x427f2fe9ca5ff669,
    0x42808dda78dcf681, 0x428192e59eed58a4, 0x4282a8053135ed05, 0x4283ce36c0026ee5,
    0x428506877a174e6d, 0x4286521522fe0042, 0x4287b20f187d339b, 0x428927b7682c2bd2,
    0x428ab463f61f3951, 0x428c597fb5baeee9, 0x428e188bf5cc501e, 0x428ff321c115d44b,
    0x4290f579aa49e96b, 0x429200e6d66765b7, 0x42931ccd15d38aa4, 0x42944a302c0df299,
    0x42958a23dcd3b012, 0x4296ddcce8715650, 0x42984662179fe02e, 0x4299c52d57e18b98,
    0x429b5b8ce962db3c, 0x429d0af49f73f361, 0x429ed4ef34bf8df7, 0x42a05d8fda3b67f6,
    0x42a15fa17cde3b94, 0x42a27198a35719da, 0x42a3946ffa404d5b, 0x42a4c9319f1c8975,
    0x42a610f813d4ba59, 0x42a76cef41357974, 0x42a8de558958a9fb, 0x42aa667ceaf6524e,
    0x42ac06cc36a74088, 0x42adc0c057346ffd, 0x42af95edae1f975a, 0x42b0c400c2d1e2f6,
    0x42b1cc61ccc04e04,
)
_C513_DAUGHTER_GRID = np.asarray(_C513_DAUGHTER_GRID_BITS, dtype="<u8").view("<f8")
_C513_DAUGHTER_GRID.setflags(write=False)


def c513_configuration_from_log_offsets(configuration, offsets):
    """Apply dimensionless ``(log L_e,kin, log L_p,kin, log B, log R)`` offsets.

    The fixed blob-frame production grids are unchanged. Injection densities
    remain in cm^-3 s^-1 per Lorentz factor, field and radius remain in G and
    cm, and ``L_e,kin``/``L_p,kin`` retain the builder's kinetic-energy
    normalization ``(gamma - 1) m c^2``. Lorentz factor remains total particle
    energy divided by rest energy. The production escape prescription remains
    ``t_esc = R/c``.
    Adiabatic rates are unchanged because the C513 scenario sets them to exact
    zero independently of radius.
    """
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    if offsets.shape != (4,):
        raise ValueError("offsets must have shape (4,)")
    electron_scale, proton_scale, magnetic_scale, radius_scale = jnp.exp(offsets)
    inverse_volume_scale = radius_scale**-3
    inverse_radius_scale = radius_scale**-1
    dynamic = {
        **configuration,
        "primary_electron_injection_dgamma_cm3_s": (
            electron_scale
            * inverse_volume_scale
            * configuration["primary_electron_injection_dgamma_cm3_s"]
        ),
        "primary_proton_injection_dgamma_cm3_s": (
            proton_scale
            * inverse_volume_scale
            * configuration["primary_proton_injection_dgamma_cm3_s"]
        ),
        "magnetic_field_G": magnetic_scale * configuration["magnetic_field_G"],
        "radius_cm": radius_scale * configuration["radius_cm"],
        "electron_escape_rate_s_inv": (
            inverse_radius_scale * configuration["electron_escape_rate_s_inv"]
        ),
        "proton_escape_rate_s_inv": (
            inverse_radius_scale * configuration["proton_escape_rate_s_inv"]
        ),
    }
    for name, scale in (
        ("electron_injection_normalization_cm3_s", electron_scale),
        ("proton_injection_normalization_cm3_s", proton_scale),
    ):
        if name in configuration:
            dynamic[name] = scale * inverse_volume_scale * configuration[name]
    return dynamic


def _cutoff_injection_nominal_integral_jax(index, cutoff_gamma):
    index = jnp.asarray(index, dtype=jnp.float64)
    cutoff = jnp.asarray(cutoff_gamma, dtype=jnp.float64)
    upper = jnp.log(cutoff)
    log_gamma = (
        0.5
        * (jnp.asarray(_INJECTION_QUADRATURE_NODES, dtype=jnp.float64) + 1.0)
        * upper
    )
    gamma = jnp.exp(log_gamma)
    integrand = (gamma - 1.0) * gamma ** (1.0 - index) * jnp.exp(-gamma / cutoff)
    return (
        0.5
        * upper
        * jnp.sum(
            jnp.asarray(_INJECTION_QUADRATURE_WEIGHTS, dtype=jnp.float64) * integrand
        )
    )


def _rescaled_cutoff_injection(configuration, particle, index, cutoff):
    baseline_index = configuration[f"{particle}_index"]
    baseline_cutoff = configuration[f"{particle}_gamma_cutoff"]
    baseline_normalization = configuration[f"{particle}_injection_normalization_cm3_s"]
    grid_name = "electron_gamma" if particle == "electron" else "proton_gamma"
    grid = configuration[grid_name]
    normalization = baseline_normalization * (
        _cutoff_injection_nominal_integral_jax(baseline_index, baseline_cutoff)
        / _cutoff_injection_nominal_integral_jax(index, cutoff)
    )
    return (
        normalization * grid ** (-index) * jnp.exp(-grid / cutoff),
        normalization,
    )


def c513_configuration_from_injection_shape_offsets(configuration, offsets):
    """Apply ``(Delta p_e, Delta p_p, Delta log gamma_e,max, Delta log gamma_p,max)``.

    Offsets are finite dimensionless local perturbations. The C513 blob-frame
    particle grids remain fixed. Both injections retain the production smooth
    exponential cutoff and the nominal luminosity integral from gamma=1 to
    gamma_max; injection arrays remain in cm^-3 s^-1 per Lorentz factor.
    """
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    if offsets.shape != (4,):
        raise ValueError("offsets must have shape (4,)")
    electron_index = configuration["electron_index"] + offsets[0]
    proton_index = configuration["proton_index"] + offsets[1]
    electron_cutoff = configuration["electron_gamma_cutoff"] * jnp.exp(offsets[2])
    proton_cutoff = configuration["proton_gamma_cutoff"] * jnp.exp(offsets[3])
    electron_injection, electron_normalization = _rescaled_cutoff_injection(
        configuration,
        "electron",
        electron_index,
        electron_cutoff,
    )
    proton_injection, proton_normalization = _rescaled_cutoff_injection(
        configuration,
        "proton",
        proton_index,
        proton_cutoff,
    )
    return {
        **configuration,
        "electron_index": electron_index,
        "proton_index": proton_index,
        "electron_gamma_cutoff": electron_cutoff,
        "proton_gamma_cutoff": proton_cutoff,
        "electron_injection_normalization_cm3_s": electron_normalization,
        "proton_injection_normalization_cm3_s": proton_normalization,
        "primary_electron_injection_dgamma_cm3_s": electron_injection,
        "primary_proton_injection_dgamma_cm3_s": proton_injection,
    }


def c513_configuration_from_escape_rate_offset(configuration, offsets):
    """Apply one shared dimensionless ``Delta log f_esc`` offset.

    The blob-comoving C513 escape sinks remain in s^-1 and follow
    ``lambda_e,esc = lambda_p,esc = f_esc c/R`` around the production
    baseline ``f_esc = 1``. Electron and proton rates share the same factor,
    as in the existing builder; all particle and photon grids remain fixed.
    Finite local offsets are valid while the resulting float64 rates remain
    finite and positive. No injection, cooling, interaction, or frame input is
    changed.
    """
    offsets = jnp.asarray(offsets, dtype=jnp.float64)
    if offsets.shape != (1,):
        raise ValueError("offsets must have shape (1,)")
    scale = jnp.exp(offsets[0])
    return {
        **configuration,
        "electron_escape_rate_s_inv": (
            scale * configuration["electron_escape_rate_s_inv"]
        ),
        "proton_escape_rate_s_inv": (scale * configuration["proton_escape_rate_s_inv"]),
    }


def _integral(values, coordinate):
    return float(np.trapezoid(np.asarray(values), x=np.asarray(coordinate)))


def _target_grid():
    grid = np.geomspace(1.0e-9, 1.0e6, 96) * EV_TO_ERG
    points_per_decade = (grid.size - 1) / np.log10(grid[-1] / grid[0])
    for maximum in (6.1e-2, 6.1e-1, 6.1):
        intervals = int(round(points_per_decade * np.log10(maximum / grid[-1])))
        grid = np.concatenate(
            (grid, np.geomspace(grid[-1], maximum, intervals + 1)[1:])
        )
    if grid.size != 137 or grid[-1] != 6.1:
        raise RuntimeError(
            "C overlap target grid no longer has 137 nodes through 6.1 erg"
        )
    return grid


def _aligned_lepton_grid(gamma_ray_energy):
    maximum = (gamma_ray_energy[-1] + 1.0e6 * EV_TO_ERG) / _MEC2_CGS
    f4097 = np.geomspace(1.000001, maximum, 4097)
    if _C513_DAUGHTER_GRID[0] != 1.000001 or _C513_DAUGHTER_GRID[-1] != maximum:
        raise RuntimeError("canonical C513 endpoints no longer match the model")
    f4097[::8] = _C513_DAUGHTER_GRID
    c2049 = f4097[::2]
    c513 = f4097[::8]
    if not (
        c513.size == 513
        and np.array_equal(c513, c2049[::4])
        and np.array_equal(c513[[0, -1]], f4097[[0, -1]])
    ):
        raise RuntimeError("C513 is not the exact nested C/F aligned grid")
    return c513, c2049, f4097


def _cutoff_injection(
    gamma_total_energy_over_rest_energy,
    cutoff,
    kinetic_injection_luminosity_erg_s,
    rest_energy_erg,
    volume_cm3,
):
    """Normalize ``Q(gamma)`` by kinetic energy without changing its shape."""

    grid = gamma_total_energy_over_rest_energy

    def logarithmic_integrand(log_gamma):
        gamma = np.exp(log_gamma)
        return (gamma - 1.0) / gamma * np.exp(-gamma / cutoff)

    nominal_integral = quad(
        logarithmic_integrand,
        0.0,
        np.log(cutoff),
        epsabs=0.0,
        epsrel=1.0e-12,
        limit=200,
    )[0]
    normalization = kinetic_injection_luminosity_erg_s / (
        volume_cm3 * rest_energy_erg * nominal_integral
    )
    injection = normalization * grid**-2.0 * np.exp(-grid / cutoff)
    recovered = volume_cm3 * rest_energy_erg * normalization * nominal_integral
    continuous_full = (
        volume_cm3
        * rest_energy_erg
        * normalization
        * quad(
            logarithmic_integrand,
            0.0,
            np.log(grid[-1]),
            epsabs=0.0,
            epsrel=1.0e-12,
            limit=300,
        )[0]
    )
    discrete_full = (
        volume_cm3
        * rest_energy_erg
        * _integral(
            grid * (grid - 1.0) * injection,
            np.log(grid),
        )
    )
    return injection, {
        "requested_kinetic_injection_luminosity_erg_s": (
            kinetic_injection_luminosity_erg_s
        ),
        "normalization_cm3_s": normalization,
        "recovered_nominal_kinetic_injection_luminosity_erg_s": recovered,
        "continuous_full_grid_kinetic_injection_luminosity_erg_s": continuous_full,
        "discrete_full_grid_kinetic_injection_luminosity_erg_s": discrete_full,
        "tail_above_cutoff_kinetic_injection_luminosity_erg_s": (
            continuous_full - recovered
        ),
        "finite_grid_kinetic_injection_quadrature_difference_erg_s": (
            discrete_full - continuous_full
        ),
        "normalization_upper_gamma": cutoff,
        "evaluated_grid_upper_gamma": float(grid[-1]),
    }


def _grid_metadata(grid, source):
    grid = np.asarray(grid)
    return {
        "nodes": int(grid.size),
        "minimum": float(grid[0]),
        "maximum": float(grid[-1]),
        "dtype": str(grid.dtype),
        "source": source,
    }


def build_c513_configuration():
    """Build validated blob-comoving C513 quantities and numerical grids.

    Particle ``gamma`` is total energy divided by rest energy. Particle
    injections and densities are per unit gamma; photon and neutrino fields
    and sources are per unit energy. Electron/proton injection luminosities
    are unchanged kinetic-energy integrals weighted by ``(gamma - 1) m c^2``.
    The fixed external photon state is already blob-comoving; this builder
    performs no host/AGN-to-blob transformation.
    """
    split_energy = 1.0e6 * EV_TO_ERG
    gamma_ray_energy = np.sort(
        np.append(np.geomspace(1.0e2, 1.0e19, 112) * EV_TO_ERG, split_energy)
    )
    synchrotron_energy = _target_grid()
    electron_gamma, c2049, f4097 = _aligned_lepton_grid(gamma_ray_energy)

    # Match the validated 40 intervals/decade proton resolution while extending
    # the existing 1e5..1e11 range down to the benchmark gamma_min=1 boundary.
    proton_gamma = np.geomspace(1.000001, 1.0e11, 441)
    pion_gamma = np.geomspace(1.0e3, 3.0e11, 2400)
    muon_gamma = np.geomspace(1.0e2, 4.0e11, 2700)
    neutrino_energy = np.geomspace(1.0e-11, 7.0e7, 2400)
    external_energy = np.geomspace(1.0e-4, 1.0e2, 64) * EV_TO_ERG
    volume = 4.0 * np.pi * RADIUS_CM**3 / 3.0

    electron_injection, electron_audit = _cutoff_injection(
        electron_gamma,
        ELECTRON_CUTOFF_GAMMA,
        ELECTRON_KINETIC_INJECTION_LUMINOSITY_ERG_S,
        _MEC2_CGS,
        volume,
    )
    proton_injection, proton_audit = _cutoff_injection(
        proton_gamma,
        PROTON_CUTOFF_GAMMA,
        PROTON_KINETIC_INJECTION_LUMINOSITY_ERG_S,
        _PROTON_REST_ENERGY_CGS,
        volume,
    )
    escape_rate = _C_CGS / RADIUS_CM
    initial_internal = {
        "low_energy_erg": jnp.asarray(synchrotron_energy),
        "low_density_cm3_erg": jnp.zeros(synchrotron_energy.size, dtype=jnp.float64),
        "high_energy_erg": jnp.asarray(gamma_ray_energy),
        "high_density_cm3_erg": jnp.zeros(gamma_ray_energy.size, dtype=jnp.float64),
    }
    fixed_external = {
        "energy_erg": jnp.asarray(external_energy),
        "density_cm3_erg": jnp.zeros(external_energy.size, dtype=jnp.float64),
    }
    production = {
        "proton_gamma": jnp.asarray(proton_gamma),
        "primary_proton_injection_dgamma_cm3_s": jnp.asarray(proton_injection),
        "pion_gamma": jnp.asarray(pion_gamma),
        "muon_gamma": jnp.asarray(muon_gamma),
        "synchrotron_photon_energy_erg": jnp.asarray(synchrotron_energy),
        "gamma_ray_energy_erg": jnp.asarray(gamma_ray_energy),
        "electron_gamma": jnp.asarray(electron_gamma),
        "neutrino_energy_erg": jnp.asarray(neutrino_energy),
        "primary_electron_injection_dgamma_cm3_s": jnp.asarray(electron_injection),
        "primary_positron_injection_dgamma_cm3_s": jnp.zeros(
            electron_gamma.size, dtype=jnp.float64
        ),
        "electron_index": jnp.float64(2.0),
        "proton_index": jnp.float64(2.0),
        "electron_gamma_cutoff": jnp.float64(ELECTRON_CUTOFF_GAMMA),
        "proton_gamma_cutoff": jnp.float64(PROTON_CUTOFF_GAMMA),
        "electron_injection_normalization_cm3_s": jnp.float64(
            electron_audit["normalization_cm3_s"]
        ),
        "proton_injection_normalization_cm3_s": jnp.float64(
            proton_audit["normalization_cm3_s"]
        ),
        "magnetic_field_G": jnp.float64(MAGNETIC_FIELD_G),
        "radius_cm": jnp.float64(RADIUS_CM),
        "proton_adiabatic_rate_s_inv": jnp.float64(0.0),
        "proton_escape_rate_s_inv": jnp.float64(escape_rate),
        "fractional_bethe_heitler_energy_loss_rate_s_inv": jnp.zeros(
            proton_gamma.size, dtype=jnp.float64
        ),
        "electron_adiabatic_rate_s_inv": jnp.float64(0.0),
        "electron_escape_rate_s_inv": jnp.float64(escape_rate),
        "synchrotron_split_energy_erg": jnp.float64(split_energy),
        "additional_fixed_inverse_compton_loss_rate_dgamma_s": jnp.float64(0.0),
        "additional_continuous_loss_rate_dgamma_s": jnp.float64(0.0),
        "include_external_gamma_gamma_target": False,
        "pair_daughter_gamma": None,
        "inner_relaxation_omega": 1.0,
        "inner_relative_tolerance": INNER_RTOL,
        "inner_maximum_iterations": INNER_MAXIMUM_ITERATIONS,
        "bethe_heitler_mode": "dynamic",
        "enable_ssc": True,
        "enable_internal_gamma_gamma": True,
        "enable_ec": False,
        "enable_pion_synchrotron_photons": True,
        "enable_muon_synchrotron_photons": True,
    }
    metadata = {
        "scenario_name": SCENARIO_NAME,
        "reference": REFERENCE,
        "reference_url": REFERENCE_URL,
        "physical_parameters": {
            "radius_cm": RADIUS_CM,
            "magnetic_field_G": MAGNETIC_FIELD_G,
            "doppler_factor": DOPPLER_FACTOR,
            "external_photon_field_blob": "none",
            "electron_gamma_min": 1.0,
            "electron_gamma_cutoff": ELECTRON_CUTOFF_GAMMA,
            "electron_index": 2.0,
            "electron_kinetic_injection_luminosity_erg_s": (
                ELECTRON_KINETIC_INJECTION_LUMINOSITY_ERG_S
            ),
            "electron_escape_time_s": RADIUS_CM / _C_CGS,
            "proton_gamma_min": 1.0,
            "proton_gamma_cutoff": PROTON_CUTOFF_GAMMA,
            "proton_index": 2.0,
            "proton_kinetic_injection_luminosity_erg_s": (
                PROTON_KINETIC_INJECTION_LUMINOSITY_ERG_S
            ),
            "proton_escape_time_s": RADIUS_CM / _C_CGS,
            "primary_positron_injection": "exact_zero",
            "electron_adiabatic_loss": "exact_zero",
            "proton_adiabatic_loss": "exact_zero",
        },
        "quantity_contracts": {
            "particle_gamma": "total particle energy divided by rest energy",
            "particle_injection": "Q(gamma), cm^-3 s^-1 per unit gamma",
            "particle_density": "N(gamma), cm^-3 per unit gamma",
            "injection_luminosity": (
                "kinetic: V m c^2 integral[(gamma - 1) Q(gamma) dgamma]"
            ),
            "photon_neutrino_measure": "per energy, never per gamma",
            "fixed_external_photon_frame": (
                "blob-comoving; host/AGN fields require an explicit upstream transform"
            ),
        },
        "injection_audits": {
            "electron": electron_audit,
            "proton": proton_audit,
        },
        "grids": {
            "electron_lepton_transport": _grid_metadata(
                electron_gamma, "C513=F4097[::8]=C2049[::4]"
            ),
            "pair_daughter": _grid_metadata(
                electron_gamma, "native identity path, pair_daughter_gamma=None"
            ),
            "proton": _grid_metadata(
                proton_gamma,
                "photomeson_decay_chain 1e11 upper range at 40 intervals/decade",
            ),
            "pion": _grid_metadata(pion_gamma, "examples/photomeson_decay_chain.py"),
            "muon": _grid_metadata(muon_gamma, "examples/photomeson_decay_chain.py"),
            "neutrino": _grid_metadata(
                neutrino_energy, "examples/photomeson_decay_chain.py"
            ),
            "internal_low_photon": _grid_metadata(
                synchrotron_energy, "validated overlap target configuration C"
            ),
            "high_energy_photon": _grid_metadata(
                gamma_ray_energy,
                "aligned C2049/F4097 EM configuration plus exact 1 MeV split",
            ),
            "external_zero_placeholder": _grid_metadata(
                external_energy, "historical 64-node external grid, density zero"
            ),
        },
        "grid_checks": {
            "c513_exactly_513": electron_gamma.size == 513,
            "c513_equals_c2049_every_four": np.array_equal(electron_gamma, c2049[::4]),
            "c513_equals_f4097_every_eight": np.array_equal(electron_gamma, f4097[::8]),
            "native_pair_grid": production["pair_daughter_gamma"] is None,
            "all_float64": all(
                np.asarray(value).dtype == np.float64
                for value in (
                    electron_gamma,
                    proton_gamma,
                    pion_gamma,
                    muon_gamma,
                    neutrino_energy,
                    synchrotron_energy,
                    gamma_ray_energy,
                )
            ),
        },
        "numerical_strategy": {
            "outer_method": "direct_picard",
            "outer_rtol": OUTER_RTOL,
            "max_outer_iterations": MAX_OUTER_ITERATIONS,
            "required_consecutive_passes": REQUIRED_CONSECUTIVE_PASSES,
            "inner_relaxation_omega": 1.0,
            "inner_relative_tolerance": INNER_RTOL,
            "inner_maximum_iterations": INNER_MAXIMUM_ITERATIONS,
            "parameter_scan": False,
            "outer_mixing": False,
        },
        "observer_output": {
            "doppler_factor": DOPPLER_FACTOR,
            "redshift": "not supplied; no cosmological shift applied",
            "luminosity_distance": "not supplied; no Earth flux produced",
        },
    }
    return {
        "initial_internal_photon_state_blob": initial_internal,
        "fixed_external_photon_state_blob": fixed_external,
        "production_configuration": production,
        "metadata": metadata,
    }


def prepare_c513_responses(setup, *, cache_dir, allow_build=False):
    """Return fixed C513 BH and gamma-gamma responses for the internal solver.

    ``setup`` is returned by :func:`build_c513_configuration`.
    ``cache_dir`` selects the existing strict persistent BH cache;
    ``allow_build=False`` requires exact cache hits. Response units, grids,
    compatibility checks, and physical assumptions are inherited unchanged
    from the existing response builders.
    """
    initial = setup["initial_internal_photon_state_blob"]
    external = setup["fixed_external_photon_state_blob"]
    configuration = setup["production_configuration"]
    components = _target_components(
        initial,
        external,
        configuration["synchrotron_split_energy_erg"],
    )
    responses = {
        name: load_or_prepare_bethe_heitler_pair_response_jax(
            configuration["electron_gamma"],
            configuration["proton_gamma"],
            component["energy_erg"],
            cache_dir=cache_dir,
            allow_build=allow_build,
        )
        for name, component in components.items()
    }
    fingerprints = {
        name: bethe_heitler_response_fingerprint(
            configuration["electron_gamma"],
            configuration["proton_gamma"],
            component["energy_erg"],
        )
        for name, component in components.items()
    }
    pair_gamma = configuration.get("pair_daughter_gamma")
    if pair_gamma is None:
        pair_gamma = configuration["electron_gamma"]
    gamma_gamma_response = prepare_gamma_gamma_pair_response_jax(
        pair_gamma,
        configuration["gamma_ray_energy_erg"],
        configuration["synchrotron_photon_energy_erg"],
    )
    return {
        "bethe_heitler_pair_responses": responses,
        "bethe_heitler_response_fingerprints": fingerprints,
        "gamma_gamma_pair_response": gamma_gamma_response,
    }


def _canonical_broadband_photon_support(configuration):
    low_energy = np.asarray(configuration["synchrotron_photon_energy_erg"])
    high_energy = np.asarray(configuration["gamma_ray_energy_erg"])
    split = float(np.asarray(configuration["synchrotron_split_energy_erg"]))
    low_stop = int(np.searchsorted(low_energy, split, side="right"))
    high_start = int(np.searchsorted(high_energy, split, side="right"))
    broadband_energy = np.concatenate(
        (low_energy[:low_stop], high_energy[high_start:])
    )
    if np.any(np.diff(broadband_energy) <= 0.0):
        raise RuntimeError("canonical broadband photon energy grid is not ordered")
    return broadband_energy, low_stop, high_start


def _canonical_broadband_escaped_photon_sed(configuration, inner):
    broadband_energy, low_stop, high_start = _canonical_broadband_photon_support(
        configuration
    )
    broadband_escaped = jnp.concatenate(
        (
            jnp.asarray(
                inner["total_low_energy_photon_escaped_sed_erg_s"]
            )[:low_stop],
            jnp.asarray(inner["escaped_gamma_sed_erg_s"])[high_start:],
        )
    )
    return broadband_energy, broadband_escaped


def _canonical_broadband_escaped_photon_components(configuration, final):
    """Return additive escaped process groups on the canonical photon grid."""
    energy, low_stop, high_start = _canonical_broadband_photon_support(
        configuration
    )
    inner = final["inner_electromagnetic_cascade"]
    low_meson = final["meson_synchrotron_low_energy_intrinsic_seds_erg_s"]
    high_meson = final["meson_synchrotron_intrinsic_seds_erg_s"]
    low_intrinsic = {
        "transported_lepton_synchrotron": inner[
            "electron_synchrotron_intrinsic_sed_erg_s"
        ],
        "transported_lepton_ssc": inner["ssc_low_energy_intrinsic_sed_erg_s"],
        "proton_synchrotron": inner[
            "proton_synchrotron_low_energy_intrinsic_sed_erg_s"
        ],
        "neutral_pion_decay": jnp.zeros_like(
            configuration["synchrotron_photon_energy_erg"]
        ),
        "charged_pion_synchrotron": low_meson["pion_total"],
        "muon_synchrotron": low_meson["muon_total"],
    }
    high_intrinsic = {
        "transported_lepton_synchrotron": inner[
            "electron_synchrotron_high_energy_intrinsic_sed_erg_s"
        ],
        "transported_lepton_ssc": inner["ssc_high_energy_intrinsic_sed_erg_s"],
        "proton_synchrotron": inner[
            "proton_synchrotron_high_energy_intrinsic_sed_erg_s"
        ],
        "neutral_pion_decay": inner["pi0_intrinsic_sed_erg_s"],
        "charged_pion_synchrotron": high_meson["pion_total"],
        "muon_synchrotron": high_meson["muon_total"],
    }
    absorption = inner["electron_synchrotron_absorption_coefficient_cm1"]
    high_transfer = inner["gamma_gamma_transfer_factor"]
    components = {
        name: jnp.concatenate(
            (
                _transfer_from_quantities(
                    absorption, values, configuration["radius_cm"]
                )[:low_stop],
                jnp.asarray(high_intrinsic[name] * high_transfer)[high_start:],
            )
        )
        for name, values in low_intrinsic.items()
    }
    return energy, components


_SOURCE_RESOLVED_ADDITIVE_COMPONENTS = (
    "primary_electron_synchrotron",
    "primary_electron_ssc",
    "bethe_heitler_secondary_radiation",
    "photomeson_secondary_radiation",
    "gamma_gamma_reprocessed_radiation",
    "proton_synchrotron",
    "neutral_pion_decay",
    "charged_pion_synchrotron",
    "muon_synchrotron",
)


def _closure_errors(actual, expected, coordinate):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    coordinate = np.asarray(coordinate, dtype=np.float64)
    scale = float(np.max(np.abs(expected)))
    if scale == 0.0:
        return (0.0, 0.0) if np.array_equal(actual, expected) else (np.inf, np.inf)
    support = np.abs(expected) > scale * 1.0e-20
    maximum = float(
        np.max(np.abs(actual[support] - expected[support]) / np.abs(expected[support]))
    )
    denominator = float(np.trapezoid(np.abs(expected), x=coordinate))
    weighted = float(
        np.trapezoid(np.abs(actual - expected), x=coordinate) / denominator
    )
    return maximum, weighted


def _source_resolved_broadband_components(configuration, final):
    """Decompose one converged solution with its final linear lepton operator.

    The production cascade is not iterated here. Final cooling, escape, target
    photon density, SSA opacity, and gamma-gamma transfer are frozen; the same
    transport and radiation owners are then evaluated once per mutually
    exclusive injection class. Gamma-gamma pairs retain aggregate reprocessed
    ownership because their mixed parent-photon ancestry is not separable.
    """
    inner = final["inner_electromagnetic_cascade"]
    if np.any(np.asarray(inner["external_compton_intrinsic_sed_erg_s"]) != 0.0):
        raise RuntimeError(
            "source-resolved canonical decomposition requires the Public model's "
            "zero external photon field"
        )

    electron_gamma = jnp.asarray(configuration["electron_gamma"], dtype=jnp.float64)
    photomeson = final["photomeson_chain"]
    sources = {
        "primary": (
            configuration["primary_electron_injection_dgamma_cm3_s"],
            configuration["primary_positron_injection_dgamma_cm3_s"],
        ),
        "bethe_heitler": (
            final["q_bh_electron_dgamma_cm3_s"],
            final["q_bh_positron_dgamma_cm3_s"],
        ),
        "photomeson": (photomeson["electron"], photomeson["positron"]),
        "gamma_gamma": (
            inner["electron_pair_injection_dgamma_cm3_s"],
            inner["positron_pair_injection_dgamma_cm3_s"],
        ),
    }
    total_loss = jnp.asarray(
        inner["total_inverse_compton_loss_rate_dgamma_s"], dtype=jnp.float64
    ) + jnp.asarray(
        configuration.get("additional_continuous_loss_rate_dgamma_s", 0.0),
        dtype=jnp.float64,
    )
    populations = {
        name: electron_positron_steady_state_transport_jax(
            electron_gamma,
            electron_source,
            positron_source,
            configuration["magnetic_field_G"],
            configuration["electron_adiabatic_rate_s_inv"],
            configuration["electron_escape_rate_s_inv"],
            additional_continuous_loss_rate_dgamma_s=total_loss,
        )[:2]
        for name, (electron_source, positron_source) in sources.items()
    }

    low_energy = jnp.asarray(
        configuration["synchrotron_photon_energy_erg"], dtype=jnp.float64
    )
    high_energy = jnp.asarray(
        configuration["gamma_ray_energy_erg"], dtype=jnp.float64
    )
    split = jnp.asarray(configuration["synchrotron_split_energy_erg"])
    broadband_energy, low_stop, high_start = _canonical_broadband_photon_support(
        configuration
    )
    target_density = inner["internal_synchrotron_photon_density_cm3_erg"]
    ssc_enabled = bool(np.asarray(inner["enable_ssc"]))
    low_synchrotron = {}
    high_synchrotron = {}
    low_inverse_compton = {}
    high_inverse_compton = {}
    for name, (electron_density, positron_density) in populations.items():
        lepton_density = electron_density + positron_density
        low_synchrotron[name] = electron_synchrotron_sed_jax(
            low_energy,
            electron_gamma,
            lepton_density,
            magnetic_field_G=configuration["magnetic_field_G"],
            radius_cm=configuration["radius_cm"],
        )
        high_synchrotron[name] = jnp.where(
            high_energy > split,
            electron_synchrotron_sed_jax(
                high_energy,
                electron_gamma,
                lepton_density,
                magnetic_field_G=configuration["magnetic_field_G"],
                radius_cm=configuration["radius_cm"],
            ),
            0.0,
        )
        if ssc_enabled:
            low_owned = inverse_compton_sed_jax(
                low_energy[:low_stop],
                electron_gamma,
                lepton_density,
                low_energy,
                target_density,
                radius_cm=configuration["radius_cm"],
            )
            low_inverse_compton[name] = jnp.zeros_like(low_energy).at[
                :low_stop
            ].set(low_owned)
            high_inverse_compton[name] = jnp.where(
                high_energy > split,
                inverse_compton_sed_jax(
                    high_energy,
                    electron_gamma,
                    lepton_density,
                    low_energy,
                    target_density,
                    radius_cm=configuration["radius_cm"],
                ),
                0.0,
            )
        else:
            low_inverse_compton[name] = jnp.zeros_like(low_energy)
            high_inverse_compton[name] = jnp.zeros_like(high_energy)

    absorption = inner["electron_synchrotron_absorption_coefficient_cm1"]
    high_transfer = inner["gamma_gamma_transfer_factor"]

    def escaped(low_intrinsic, high_intrinsic):
        return jnp.concatenate(
            (
                _transfer_from_quantities(
                    absorption, low_intrinsic, configuration["radius_cm"]
                )[:low_stop],
                jnp.asarray(high_intrinsic * high_transfer)[high_start:],
            )
        )

    _, aggregate = _canonical_broadband_escaped_photon_components(
        configuration, final
    )
    mechanism_components = {
        "primary_electron_synchrotron": escaped(
            low_synchrotron["primary"], high_synchrotron["primary"]
        ),
        "primary_electron_inverse_compton": escaped(
            low_inverse_compton["primary"], high_inverse_compton["primary"]
        ),
        "bethe_heitler_pair_synchrotron": escaped(
            low_synchrotron["bethe_heitler"], high_synchrotron["bethe_heitler"]
        ),
        "bethe_heitler_pair_inverse_compton": escaped(
            low_inverse_compton["bethe_heitler"],
            high_inverse_compton["bethe_heitler"],
        ),
        "photomeson_secondary_pair_synchrotron": escaped(
            low_synchrotron["photomeson"], high_synchrotron["photomeson"]
        ),
        "photomeson_secondary_pair_inverse_compton": escaped(
            low_inverse_compton["photomeson"],
            high_inverse_compton["photomeson"],
        ),
        "gamma_gamma_reprocessed_pair_synchrotron": escaped(
            low_synchrotron["gamma_gamma"], high_synchrotron["gamma_gamma"]
        ),
        "gamma_gamma_reprocessed_pair_inverse_compton": escaped(
            low_inverse_compton["gamma_gamma"],
            high_inverse_compton["gamma_gamma"],
        ),
    }
    components = {
        "primary_electron_synchrotron": mechanism_components[
            "primary_electron_synchrotron"
        ],
        # Compatibility name retained; the final synchrotron target includes
        # every enabled synchrotron source, so new displays should use the
        # mechanism-accurate inverse-Compton alias below.
        "primary_electron_ssc": mechanism_components[
            "primary_electron_inverse_compton"
        ],
        "bethe_heitler_secondary_radiation": escaped(
            low_synchrotron["bethe_heitler"]
            + low_inverse_compton["bethe_heitler"],
            high_synchrotron["bethe_heitler"]
            + high_inverse_compton["bethe_heitler"],
        ),
        "photomeson_secondary_radiation": escaped(
            low_synchrotron["photomeson"] + low_inverse_compton["photomeson"],
            high_synchrotron["photomeson"] + high_inverse_compton["photomeson"],
        ),
        "gamma_gamma_reprocessed_radiation": escaped(
            low_synchrotron["gamma_gamma"] + low_inverse_compton["gamma_gamma"],
            high_synchrotron["gamma_gamma"]
            + high_inverse_compton["gamma_gamma"],
        ),
        **{
            name: aggregate[name]
            for name in (
                "proton_synchrotron",
                "neutral_pion_decay",
                "charged_pion_synchrotron",
                "muon_synchrotron",
            )
        },
        **mechanism_components,
    }

    electron_total = sum(population[0] for population in populations.values())
    positron_total = sum(population[1] for population in populations.values())
    synchrotron_actual = jnp.concatenate(
        (
            sum(low_synchrotron.values())[:low_stop],
            sum(high_synchrotron.values())[high_start:],
        )
    )
    synchrotron_expected = jnp.concatenate(
        (
            inner["electron_synchrotron_intrinsic_sed_erg_s"][:low_stop],
            inner["electron_synchrotron_high_energy_intrinsic_sed_erg_s"][
                high_start:
            ],
        )
    )
    inverse_compton_actual = jnp.concatenate(
        (
            sum(low_inverse_compton.values())[:low_stop],
            sum(high_inverse_compton.values())[high_start:],
        )
    )
    inverse_compton_expected = jnp.concatenate(
        (
            inner["ssc_low_energy_intrinsic_sed_erg_s"][:low_stop],
            inner["ssc_high_energy_intrinsic_sed_erg_s"][high_start:],
        )
    )
    _, total_escaped = _canonical_broadband_escaped_photon_sed(
        configuration, inner
    )
    metrics = {}
    for prefix, actual, expected, coordinate in (
        (
            "particle_electron",
            electron_total,
            inner["electron_density_dgamma_cm3"],
            electron_gamma,
        ),
        (
            "particle_positron",
            positron_total,
            inner["positron_density_dgamma_cm3"],
            electron_gamma,
        ),
        (
            "synchrotron",
            synchrotron_actual,
            synchrotron_expected,
            np.log(broadband_energy),
        ),
        (
            "inverse_compton",
            inverse_compton_actual,
            inverse_compton_expected,
            np.log(broadband_energy),
        ),
        (
            "escaped_blob",
            sum(components[name] for name in _SOURCE_RESOLVED_ADDITIVE_COMPONENTS),
            total_escaped,
            np.log(broadband_energy),
        ),
    ):
        maximum, weighted = _closure_errors(actual, expected, coordinate)
        metrics[f"source_resolved_{prefix}_maximum_relative_error"] = maximum
        metrics[f"source_resolved_{prefix}_weighted_error"] = weighted
    metrics["source_resolved_gamma_gamma_parent_ownership"] = "aggregate"
    return components, metrics


def run_c513_forward(
    *,
    observer_options,
    setup=None,
    production_configuration=None,
    prepared_responses=None,
    cache_dir=None,
    allow_response_build=False,
    solver_options=None,
):
    """Run the canonical internal C513 source-to-Earth full forward.

    setup defaults to build_c513_configuration.
    production_configuration may replace its configuration while retaining the
    scenario states and metadata. Fixed responses may be supplied from
    prepare_c513_responses; otherwise cache_dir is required and they
    are prepared once. solver_options overrides the setup numerical strategy,
    and observer_options is passed unchanged to with_observer_propagation.

    Input units and the blob-comoving frame are inherited from the existing
    builder, solver, and propagation functions. ``canonical_quantities`` is
    the supported quantity-aware handoff: every key names its frame, measure,
    and ``E L_E`` luminosity versus ``E F_E`` flux semantics. Its broadband
    photon fields concatenate the escaped low-energy photon segment through
    the canonical split with the escaped high-energy gamma segment strictly
    above it, without interpolation. The returned
    mapping is the complete solver result with its comoving particles,
    components, ledgers, diagnostics, and convergence state, plus the existing
    observer gamma/EBL and source/Earth neutrino outputs. No new radiation or
    transfer physics is evaluated in this wrapper.
    """
    setup = build_c513_configuration() if setup is None else setup
    if production_configuration is not None:
        setup = {**setup, "production_configuration": production_configuration}
    configuration = setup["production_configuration"]
    if prepared_responses is None:
        if cache_dir is None:
            raise ValueError(
                "cache_dir is required when prepared_responses is not supplied"
            )
        prepared_responses = prepare_c513_responses(
            setup,
            cache_dir=cache_dir,
            allow_build=allow_response_build,
        )

    strategy = setup["metadata"]["numerical_strategy"]
    options = {
        "max_outer_iterations": strategy["max_outer_iterations"],
        "outer_rtol": strategy["outer_rtol"],
        "required_consecutive_passes": strategy["required_consecutive_passes"],
    }
    options.update(solver_options or {})
    result = solve_steady_state_hadronic_cascade(
        setup["initial_internal_photon_state_blob"],
        setup["fixed_external_photon_state_blob"],
        configuration,
        bethe_heitler_pair_responses=prepared_responses["bethe_heitler_pair_responses"],
        gamma_gamma_pair_response=prepared_responses["gamma_gamma_pair_response"],
        **options,
    )
    propagated = with_observer_propagation(result, configuration, **observer_options)
    final = propagated["iterations"][-1]
    inner = final["inner_electromagnetic_cascade"]
    observer = propagated["observer_propagation"]
    broadband_energy, broadband_escaped = _canonical_broadband_escaped_photon_sed(
        configuration, inner
    )
    broadband_escaped = np.asarray(broadband_escaped)
    broadband_observer_energy, broadband_observer_luminosity = transform_comoving_sed(
        broadband_energy,
        broadband_escaped,
        doppler_factor=observer["doppler_factor"],
        redshift=observer["redshift"],
    )
    broadband_observer_flux = sed_luminosity_to_energy_flux(
        broadband_observer_luminosity,
        luminosity_distance_cm=observer["luminosity_distance_cm"],
    )
    _, broadband_components = _canonical_broadband_escaped_photon_components(
        configuration, final
    )
    source_resolved_components = {}
    source_resolved_diagnostics = {}
    if (
        not bool(np.asarray(configuration.get("enable_pp", False)))
        and "electron_pair_injection_dgamma_cm3_s" in inner
        and "photomeson_chain" in final
    ):
        source_resolved_components, source_resolved_diagnostics = (
            _source_resolved_broadband_components(configuration, final)
        )
        broadband_components = {
            **broadband_components,
            **source_resolved_components,
        }
    propagated["canonical_quantities"] = {
        "particles": {
            "electron_gamma_total_energy_over_rest_energy": configuration[
                "electron_gamma"
            ],
            "electron_primary_injection_blob_dgamma_cm3_s": configuration[
                "primary_electron_injection_dgamma_cm3_s"
            ],
            "electron_density_blob_dgamma_cm3": inner["electron_density_dgamma_cm3"],
            "positron_gamma_total_energy_over_rest_energy": configuration[
                "electron_gamma"
            ],
            "positron_primary_injection_blob_dgamma_cm3_s": configuration[
                "primary_positron_injection_dgamma_cm3_s"
            ],
            "positron_density_blob_dgamma_cm3": inner["positron_density_dgamma_cm3"],
            "proton_gamma_total_energy_over_rest_energy": configuration["proton_gamma"],
            "proton_primary_injection_blob_dgamma_cm3_s": configuration[
                "primary_proton_injection_dgamma_cm3_s"
            ],
            "proton_density_blob_dgamma_cm3": final["proton_density_dgamma_cm3"],
        },
        "photon": {
            "energy_blob_erg": configuration["gamma_ray_energy_erg"],
            "escaped_energy_luminosity_blob_erg_s": inner["escaped_gamma_sed_erg_s"],
            "energy_observer_erg": observer["gamma"]["observer_energy_erg"],
            "intrinsic_energy_flux_observer_erg_cm2_s": observer["gamma"][
                "observer_intrinsic_energy_flux_erg_cm2_s"
            ],
            "ebl_attenuated_energy_flux_earth_erg_cm2_s": observer["gamma"][
                "earth_ebl_attenuated_energy_flux_erg_cm2_s"
            ],
            "broadband_energy_blob_erg": broadband_energy,
            "broadband_escaped_energy_luminosity_blob_erg_s": (broadband_escaped),
            "broadband_energy_observer_erg": broadband_observer_energy,
            "broadband_intrinsic_energy_flux_observer_erg_cm2_s": (
                broadband_observer_flux
            ),
        },
        "photon_components": broadband_components,
        "source_resolved_photon_component_diagnostics": (
            source_resolved_diagnostics
        ),
        "neutrino": {
            "energy_blob_erg": configuration["neutrino_energy_erg"],
            "source_injection_blob_denergy_cm3_s": observer["neutrino"][
                "source_injection_blob_denergy_cm3_s"
            ],
            "energy_observer_erg": observer["neutrino"]["observer_energy_erg"],
            "unmixed_flavor_energy_flux_observer_erg_cm2_s": observer["neutrino"][
                "observer_unmixed_flavor_energy_flux_erg_cm2_s"
            ],
            "mixed_flavor_energy_flux_earth_erg_cm2_s": observer["neutrino"][
                "earth_mixed_flavor_energy_flux_erg_cm2_s"
            ],
        },
    }
    return propagated
